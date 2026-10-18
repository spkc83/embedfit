"""Few-shot, balanced, pair, and hard-negative samplers, and the batch-sampler registry."""

from embedfit.sampling.balanced import BalancedClassBatchSampler, batch_shape
from embedfit.sampling.few_shot import sample_few_shot
from embedfit.sampling.pairs import ImagePair, PairSampler, RelationshipRule
from embedfit.sampling.registry import (
    BatchSampler,
    available_samplers,
    create_sampler,
    register_sampler,
)

__all__ = [
    "BalancedClassBatchSampler",
    "BatchSampler",
    "ImagePair",
    "PairSampler",
    "RelationshipRule",
    "available_samplers",
    "batch_shape",
    "create_sampler",
    "register_sampler",
    "sample_few_shot",
]
