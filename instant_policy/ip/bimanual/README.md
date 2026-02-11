# Bimanual Reboot

This folder is the clean bimanual restart for Instant Policy style learning.

## Locked Design

Primary model inputs are relative/local:
- `P^L = (T_W_L)^-1 P^W`
- `P^R = (T_W_R)^-1 P^W`
- `T_L_R = (T_W_L)^-1 T_W_R`
- per-arm gripper state scalars

Primary model targets are relative:
- `DeltaT_L`
- `DeltaT_R`

No broad raw world-frame pose channels should be exposed as model shortcuts.

## Files

- `contracts.py`
  - strict typed contracts for observation and training targets
  - shape checks to enforce frame/data consistency
- `frame_ops.py`
  - SE(3) compose/invert/relative utilities
  - world->local point conversion
  - global-relabel invariance check helper

## Staged Build Plan

1. M0: contracts + frame ops + invariance check utilities (done)
2. M1: bimanual graph representation with per-arm local subgraphs + cross-arm edges (done: `graph_rep.py`)
3. M2: bimanual model and diffusion wrapper with relative labels only (backbone scaffold done: `model.py`)
4. M3: dataset adapter that emits contract-compliant batches (world->relative adapter done: `data_adapter.py`)
5. M4: train/eval entrypoints and smoke tests (training scaffold done: `diffusion.py`)
