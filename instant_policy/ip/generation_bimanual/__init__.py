"""Bimanual pseudo-demo generation for Instant Policy pretraining.

Implements RLBench2-inspired task families via reusable kinematic primitives.
"""

from .config import (
    BimanualGenerationConfig,
    PERACT2_BIMANUAL_TASKS,
    PERACT2_BIMANUAL_VARIATIONS,
)
from .generator import BimanualPseudoDemoGenerator

__all__ = [
    "BimanualGenerationConfig",
    "PERACT2_BIMANUAL_TASKS",
    "PERACT2_BIMANUAL_VARIATIONS",
    "BimanualPseudoDemoGenerator",
]
