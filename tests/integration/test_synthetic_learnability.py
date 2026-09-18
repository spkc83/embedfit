"""The synthetic classes must be genuinely learnable by the dummy backbone.

Later milestones rely on this to test that representation training improves embeddings,
so it is checked here with a plain cross-entropy loop.
"""

import torch
from torch import nn

from embedfit.backbones import DummyBackbone
from embedfit.testing import BAR_CLASSES, make_bar_images
from embedfit.utils import set_seed


def test_dummy_backbone_learns_bar_classes():
    set_seed(0)
    backbone = DummyBackbone(embedding_dim=16, seed=0)
    classifier = nn.Linear(16, 3)

    def tensors(seed):
        images, labels = make_bar_images(16, num_classes=3, seed=seed)
        pixels = torch.stack([backbone.preprocess(image) for image in images])
        return pixels, torch.tensor([BAR_CLASSES.index(label) for label in labels])

    train_x, train_y = tensors(seed=0)
    test_x, test_y = tensors(seed=1)
    parameters = [*backbone.parameters(), *classifier.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=0.02)
    loss_fn = nn.CrossEntropyLoss()

    backbone.train()
    initial_loss = loss_fn(classifier(backbone(train_x)), train_y).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_fn(classifier(backbone(train_x)), train_y)
        loss.backward()
        optimizer.step()

    backbone.eval()
    with torch.no_grad():
        accuracy = (classifier(backbone(test_x)).argmax(dim=1) == test_y).float().mean().item()
    assert loss.item() < 0.5 * initial_loss
    assert accuracy >= 0.9
