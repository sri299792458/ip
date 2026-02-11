"""Bimanual Instant Policy foundations.

This package contains the clean bimanual reboot:
- explicit frame contracts,
- SE(3) utilities,
- staged implementation modules.
"""

from .contracts import (
    BimanualObservation,
    BimanualTargets,
    build_observation_from_world,
)
from .frame_ops import (
    compose_transforms,
    invert_transform,
    relative_transform,
    transform_points,
    world_to_local_points,
    check_global_relabel_invariance,
)
from .graph_rep import BimanualGraphConfig, BimanualGraphRep
from .model import BimanualBackbone, BimanualModelConfig
from .data_adapter import BimanualWorldBatch, build_obs_targets, relabel_world

__all__ = [
    "BimanualObservation",
    "BimanualTargets",
    "build_observation_from_world",
    "compose_transforms",
    "invert_transform",
    "relative_transform",
    "transform_points",
    "world_to_local_points",
    "check_global_relabel_invariance",
    "BimanualGraphConfig",
    "BimanualGraphRep",
    "BimanualBackbone",
    "BimanualModelConfig",
    "BimanualWorldBatch",
    "build_obs_targets",
    "relabel_world",
]
