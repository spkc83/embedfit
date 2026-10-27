"""Selecting the normalisation-layer or bias parameters of a backbone (ADR 0021)."""

import pytest
import torch
from torch import nn

from embedfit import EmbedFitModel, TrainingArguments
from embedfit.backbones import DummyBackbone, ImageBackbone
from embedfit.exceptions import ConfigurationError


def trainable(backbone):
    return {name for name, p in backbone.named_parameters() if p.requires_grad}


class _Normed(ImageBackbone):
    """Two blocks holding every kind of normalisation layer, attention, and a final norm."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.first = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.GroupNorm(2, 4))
        self.second = nn.ModuleDict(
            {
                "attention": nn.MultiheadAttention(4, 1),
                "norm": nn.LayerNorm(4),
                "plain": nn.BatchNorm2d(4, affine=False),
                "instance": nn.InstanceNorm2d(4, affine=True),
                "scale": nn.Linear(4, 4, bias=False),
            }
        )
        self.final = nn.LayerNorm(4)

    identifier = "test/normed"
    embedding_dim = 4

    def preprocess(self, image):
        raise NotImplementedError

    def forward_features(self, pixel_values):
        features = self.first(self.stem(pixel_values))
        features = self.second["instance"](self.second["plain"](features)).mean(dim=(2, 3))
        tokens = features.unsqueeze(0)
        attended, _ = self.second["attention"](tokens, tokens, tokens)
        return self.final(self.second["scale"](self.second["norm"](attended[0])))

    def save_pretrained(self, path):
        raise NotImplementedError

    def parameter_blocks(self):
        return [self._names_of(self.first), self._names_of(self.second)]


NORMS = {
    "first.1.weight",
    "first.1.bias",
    "first.2.weight",
    "first.2.bias",
    "second.norm.weight",
    "second.norm.bias",
    "second.instance.weight",
    "second.instance.bias",
    "final.weight",
    "final.bias",
}
BIASES = {
    "stem.bias",
    "first.0.bias",
    "first.1.bias",
    "first.2.bias",
    "second.attention.in_proj_bias",
    "second.attention.out_proj.bias",
    "second.norm.bias",
    "second.instance.bias",
    "final.bias",
}


def test_layernorm_selects_the_parameters_of_every_normalisation_layer():
    backbone = _Normed()
    backbone.unfreeze(trainable_parameters="layernorm")
    assert trainable(backbone) == NORMS
    assert backbone.trainable_parameter_names(trainable_parameters="layernorm") == [
        name for name, _ in backbone.named_parameters() if name in NORMS
    ]


def test_bias_selects_every_bias_term():
    backbone = _Normed()
    backbone.unfreeze(trainable_parameters="bias")
    assert trainable(backbone) == BIASES


def test_all_is_the_whole_backbone():
    backbone = _Normed()
    backbone.freeze()
    backbone.unfreeze(trainable_parameters="all")
    assert trainable(backbone) == {name for name, _ in backbone.named_parameters()}
    backbone.train()
    assert all(module.training for module in backbone.modules())


@pytest.mark.parametrize(("kind", "selected"), [("layernorm", NORMS), ("bias", BIASES)])
def test_with_last_blocks_the_selection_is_within_those_blocks(kind, selected):
    backbone = _Normed()
    backbone.unfreeze(last_blocks=1, trainable_parameters=kind)
    # The second block and the tail (the final norm).
    in_last = {n for n, _ in backbone.named_parameters() if n.startswith(("second.", "final."))}
    assert trainable(backbone) == selected & in_last
    backbone.unfreeze(last_blocks=2, trainable_parameters=kind)
    assert trainable(backbone) == selected - {"stem.bias"}


def test_a_selection_without_parameters_is_an_error_and_changes_nothing(dummy_backbone):
    dummy_backbone.freeze()
    with pytest.raises(ConfigurationError, match="'layernorm' selects no parameter.*dummy/tiny"):
        dummy_backbone.unfreeze(trainable_parameters="layernorm")
    assert trainable(dummy_backbone) == set() and dummy_backbone.is_frozen
    backbone = DummyBackbone(seed=0)
    backbone.features.insert(1, nn.GroupNorm(2, 8))  # The second of four blocks.
    with pytest.raises(ConfigurationError, match="within its last 2 blocks"):
        backbone.unfreeze(last_blocks=2, trainable_parameters="layernorm")
    backbone.unfreeze(last_blocks=3, trainable_parameters="layernorm")
    assert trainable(backbone) == {"features.1.weight", "features.1.bias"}


def test_unknown_selections_are_rejected(dummy_backbone):
    with pytest.raises(ConfigurationError, match="trainable_parameters must be one of"):
        dummy_backbone.unfreeze(trainable_parameters="norms")


def test_batch_norm_statistics_follow_the_affine_parameters():
    backbone = _Normed()
    backbone.unfreeze(trainable_parameters="bias")
    backbone.train()
    # Batch norm with a trainable bias trains its statistics unless freeze_batch_norm holds them
    # (in training); without affine parameters it has nothing that trains and stays in eval.
    assert backbone.first[1].training
    assert not backbone.second["plain"].training
    backbone.unfreeze()
    backbone.train()
    assert backbone.second["plain"].training


def test_only_the_selected_parameters_receive_gradients(dummy_backbone):
    dummy_backbone.unfreeze(trainable_parameters="bias")
    dummy_backbone(torch.rand(2, 3, 32, 32)).sum().backward()
    for name, parameter in dummy_backbone.named_parameters():
        assert (parameter.grad is not None) == name.endswith(".bias"), name


def test_the_model_passes_the_selection_through(dummy_backbone):
    model = EmbedFitModel(dummy_backbone, labels=["a", "b"])
    model.unfreeze_backbone(1, "bias")
    assert trainable(dummy_backbone) == {"projection.bias"}
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in dummy_backbone.parameters())


def test_training_arguments_validate_the_selection():
    assert TrainingArguments().trainable_parameters == "all"
    assert TrainingArguments(trainable_parameters="bias").trainable_parameters == "bias"
    with pytest.raises(ConfigurationError, match="trainable_parameters must be one of"):
        TrainingArguments(trainable_parameters="lora")
    with pytest.raises(ConfigurationError, match="trainable_parameters only applies"):
        TrainingArguments(training_mode="frozen", trainable_parameters="layernorm")
    TrainingArguments(training_mode="frozen", trainable_parameters="all")


def test_timm_norms_are_recognised_by_type():
    timm = pytest.importorskip("timm")
    from embedfit.backbones import TimmBackbone

    backbone = TimmBackbone("resnet10t", pretrained=False)
    backbone.unfreeze(trainable_parameters="layernorm")
    expected = {
        f"model.{name}.{kind}"
        for name, module in backbone.model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        for kind in ("weight", "bias")
    }
    assert expected and trainable(backbone) == expected
    # timm's RMS norm subclasses no torch norm.
    assert backbone.is_normalization(timm.layers.RmsNorm(4))
    assert not backbone.is_normalization(nn.Linear(4, 4))


def test_transformers_norms_of_its_own_are_recognised_by_name():
    transformers = pytest.importorskip("transformers")
    from embedfit.backbones import HFBackbone

    config = transformers.ViTConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        image_size=32,
        patch_size=8,
    )
    backbone = HFBackbone("local/tiny-vit", config=config)
    backbone.unfreeze(last_blocks=1, trainable_parameters="layernorm")
    # The norms of the last layer, and the final norm (the tail).
    assert trainable(backbone) == {
        name
        for name, _ in backbone.named_parameters()
        if ".1.layernorm_" in name or name.startswith("model.layernorm.")
    }
    assert len(trainable(backbone)) == 6

    class CustomLayerNorm(nn.Module):
        pass

    assert backbone.is_normalization(CustomLayerNorm())
    assert not DummyBackbone().is_normalization(CustomLayerNorm())


def test_timm_global_response_norm_is_a_norm():
    pytest.importorskip("timm")
    from timm.layers.grn import GlobalResponseNorm

    from embedfit.backbones import TimmBackbone

    backbone = TimmBackbone("resnet10t", pretrained=False)
    assert backbone.is_normalization(GlobalResponseNorm(4))


def test_transformers_convnext_v2_grn_is_a_norm():
    transformers = pytest.importorskip("transformers")
    from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2GRN

    from embedfit.backbones import HFBackbone

    config = transformers.ViTConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        image_size=32,
        patch_size=8,
    )
    backbone = HFBackbone("local/tiny-vit", config=config)
    assert backbone.is_normalization(ConvNextV2GRN(4))
