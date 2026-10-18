import json
from dataclasses import asdict

import pytest

from embedfit import TrainingArguments
from embedfit.exceptions import ConfigurationError


def test_defaults_match_the_specification():
    args = TrainingArguments()
    assert (args.training_mode, args.num_epochs, args.batch_size, args.eval_batch_size) == (
        "supcon",
        5,
        32,
        64,
    )
    assert (args.learning_rate, args.weight_decay, args.temperature) == (2e-5, 0.01, 0.07)
    assert (args.mixed_precision, args.max_grad_norm, args.seed) == (None, 1.0, 42)
    assert (args.head, args.freeze_backbone_before_head, args.device) == (
        "logistic_regression",
        True,
        None,
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"training_mode": "magic"}, "training_mode must be one of"),
        ({"num_epochs": 0}, "num_epochs"),
        ({"batch_size": 2.0}, "batch_size"),
        ({"eval_batch_size": True}, "eval_batch_size"),
        ({"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
        ({"num_workers": -1}, "num_workers"),
        ({"seed": -1}, "seed"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"temperature": -0.1}, "temperature"),
        ({"weight_decay": -1}, "weight_decay"),
        ({"triplet_margin": -0.5}, "triplet_margin"),
        ({"warmup_ratio": 1.0}, "warmup_ratio"),
        ({"max_grad_norm": 0}, "max_grad_norm"),
        ({"mixed_precision": "fp8"}, "mixed_precision"),
        ({"head": "svm"}, "head must be one of"),
        ({"device": 0}, "device"),
        ({"deterministic": "yes"}, "deterministic"),
        ({"output_dir": ""}, "output_dir"),
        ({"unfreeze_last_blocks": 0}, "unfreeze_last_blocks"),
        ({"unfreeze_last_blocks": -1}, "unfreeze_last_blocks"),
        ({"unfreeze_last_blocks": True}, "unfreeze_last_blocks"),
        ({"unfreeze_last_blocks": 1.5}, "unfreeze_last_blocks"),
        ({"training_mode": "frozen", "unfreeze_last_blocks": 1}, "unfreeze_last_blocks"),
        ({"checkpoint_dir": ""}, "checkpoint_dir"),
        ({"checkpoint_dir": 3}, "checkpoint_dir"),
        ({"training_mode": "frozen", "checkpoint_dir": "ckpt"}, "checkpoint_dir"),
        ({"output_dir": "run", "checkpoint_dir": "run"}, "checkpoint_dir"),
        ({"output_dir": "run", "checkpoint_dir": "run/checkpoints"}, "checkpoint_dir"),
        ({"output_dir": "run/model", "checkpoint_dir": "run"}, "checkpoint_dir"),
    ],
)
def test_invalid_values_fail_at_construction(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        TrainingArguments(**kwargs)


def test_none_disables_gradient_clipping():
    assert TrainingArguments(max_grad_norm=None).max_grad_norm is None


def test_new_options_default_to_off():
    args = TrainingArguments()
    assert (args.unfreeze_last_blocks, args.checkpoint_dir) == (None, None)
    configured = TrainingArguments(unfreeze_last_blocks=2, checkpoint_dir="checkpoints")
    assert (configured.unfreeze_last_blocks, configured.checkpoint_dir) == (2, "checkpoints")


def test_arguments_are_json_serialisable():
    json.dumps(asdict(TrainingArguments()))
    json.dumps(asdict(TrainingArguments(unfreeze_last_blocks=1, checkpoint_dir="checkpoints")))
