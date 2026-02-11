# Bimanual Running Notes

## Reboot (2026-02-11)

### Decision
Restart bimanual implementation from first principles with a strict frame contract and no carry-over from the earlier prototype.

### Core Representation (Locked)
- Use arm-local scene views as primary signal:
  - `P^L = (T_W_L)^-1 P^W`
  - `P^R = (T_W_R)^-1 P^W`
- Keep explicit cross-arm relation token:
  - `T_L_R = (T_W_L)^-1 T_W_R`
- Predict relative actions per arm:
  - `DeltaT_L`, `DeltaT_R`
- Keep per-arm gripper scalar states.
- Do not add global context tokens for now (align with single-arm Instant Policy philosophy).

### Branch Structure
- `instant_policy/ip/bimanual/contracts.py`
  - Typed input/target contracts + shape validation.
- `instant_policy/ip/bimanual/frame_ops.py`
  - SE(3) composition/inversion, world->local conversion, and invariance checks.
- `instant_policy/ip/bimanual/README.md`
  - Conventions + staged implementation plan.

### Implementation Stages
1. **M0 (current)**: contracts + frame ops + invariance utilities.
2. **M1 (current)**: bimanual graph representation with local per-arm subgraphs + cross-arm edges.
3. **M2 (current)**: bimanual model heads and diffusion wrapper (relative targets only).
4. **M3 (current)**: dataset adapter with strict assertions on frame semantics.
5. **M4 (current)**: training entrypoint + smoke tests.

### Non-Negotiables
- Never feed broad raw `T_W_*` pose channels into model nodes.
- Keep a single transform naming convention (`T_A_B`: maps B-frame coords to A-frame).
- Require runtime assertions for all frame-bearing tensors.

## M1 Implemented (2026-02-11)

### Changed
- Added `instant_policy/ip/bimanual/graph_rep.py`:
  - `BimanualGraphConfig`
  - `BimanualGraphRep`
  - graph builder from `BimanualObservation` with node types:
    - `scene_left`, `scene_right`, `gripper_left`, `gripper_right`
  - edge types:
    - local: scene-scene, scene-gripper, gripper-gripper
    - cross-arm: gripper_left->gripper_right and reverse
  - cross-arm edge attrs computed using explicit `T_left_right` (and inverse) only.
- Exported graph classes from `instant_policy/ip/bimanual/__init__.py`.

### Why
- This is the minimal graph substrate needed before adding transformer + diffusion heads.
- Keeps representation fully local/relative and aligned with single-arm Instant Policy philosophy.

## M2 Backbone Scaffold Implemented (2026-02-11)

### Changed
- Added `instant_policy/ip/bimanual/model.py`:
  - `BimanualModelConfig`
  - `BimanualBackbone`
  - consumes `BimanualObservation`, builds graph via `BimanualGraphRep`, runs hetero transformer encoder, and predicts per-arm relative delta channels:
    - translation (3)
    - rotation (3)
    - gripper (1)
- Exported model classes from `instant_policy/ip/bimanual/__init__.py`.

### Why
- Provides a runnable pre-diffusion bimanual model layer on top of M1 graph construction.
- Keeps outputs directly aligned with relative-action formulation needed for diffusion training later.

## M3 World->Relative Adapter Implemented (2026-02-11)

### Changed
- Added `instant_policy/ip/bimanual/data_adapter.py`:
  - `BimanualWorldBatch` with shape/contract validation,
  - `build_obs_targets(...)`:
    - builds local observation (`P^L`, `P^R`, `T_L_R`) from world tensors,
    - builds per-arm relative targets (`DeltaT_L`, `DeltaT_R`) via
      `T_curr^-1 @ T_future`,
  - `relabel_world(...)` helper for global-frame relabeling checks.
- Exported adapter symbols from `instant_policy/ip/bimanual/__init__.py`.

### Why
- Locks the data path to first-principles relative semantics before adding diffusion training code.
- Prevents accidental world-frame leakage at the dataset boundary.

## M4 Training Scaffold Implemented (2026-02-11)

### Changed
- Added `instant_policy/ip/bimanual/diffusion.py`:
  - `BimanualTrainingConfig`
  - `BimanualGraphDiffusionScaffold` (Lightning module)
  - converts world batches -> local obs + relative targets via `build_obs_targets`,
  - supervises backbone predictions against relative deltas (`SE(3)->6D + grip`).
- Exported scaffold symbols from `instant_policy/ip/bimanual/__init__.py`.

### Why
- Establishes an executable training bridge while keeping all supervision relative/local.
- Lets us validate end-to-end data+model plumbing before adding full DDIM denoising loops.

## Smoke Script Added (2026-02-11)

### Changed
- Added `instant_policy/ip/scripts/smoke_bimanual.py`:
  - builds synthetic world batch,
  - runs one `BimanualGraphDiffusionScaffold.training_step(...)`,
  - prints `smoke_ok` and loss.
- Documented command in `instant_policy/ip/bimanual/README.md`.

### Why
- Gives a one-command sanity check for the reboot stack on any environment with the `ip` package installed.

## DDIM Upgrade (2026-02-11)

### Changed
- Replaced the previous training scaffold in `instant_policy/ip/bimanual/diffusion.py` with a full dual-arm DDIM adaptation:
  - diffusion noise injection for left/right relative actions + gripper states,
  - per-arm Instant-Policy-style label construction via `get_labels(...)`,
  - iterative denoising loop for inference (`test_step`) with rigid keypoint fit updates,
  - per-arm normalizers and optimizer scheduler parity with original Instant Policy.
- Upgraded `instant_policy/ip/bimanual/model.py` for DDIM compatibility:
  - per-node outputs `[B, P, G, 7]`,
  - diffusion timestep conditioning,
  - `get_transformed_node_pos(...)` and `get_labels(...)` helpers.
- Exported `BimanualGraphDiffusion` in `instant_policy/ip/bimanual/__init__.py`.

### Why
- Aligns bimanual training/inference mechanics with the proven single-arm Instant Policy DDIM path while preserving the local/relative bimanual data contract.

## GPU Training Entry Point (2026-02-11)

### Changed
- Added `instant_policy/ip/configs/bimanual_config.py` with default bimanual training config (GPU-first).
- Added `instant_policy/ip/train_bimanual.py`:
  - dataset loading from `ip/bimanual/dataset.py`,
  - bimanual graph/backbone/diffusion construction,
  - Lightning trainer wiring (checkpointing, optional wandb, resume/auto-resume).
- Updated `instant_policy/ip/bimanual/data_adapter.py` with `BimanualWorldBatch.to(device)`.
- Updated `instant_policy/ip/bimanual/diffusion.py` so incoming dataloader batches are moved to model device before world->relative conversion.
- Updated `instant_policy/ip/bimanual/README.md` with training command and required sample keys.

### Why
- Makes bimanual branch runnable end-to-end for real training jobs without manual notebook glue.
- Removes CPU/GPU mismatch risk from dataloader-fed batches.

## Bimanual Pseudo-Data Scaffold (2026-02-11)

### Decision
- Pretraining for bimanual should follow the same single-arm Instant Policy philosophy:
  - kinematic pseudo trajectories,
  - compositional task primitives,
  - broad synthetic pretrain then RLBench2 fine-tune.
- Use RLBench2 bimanual task taxonomy (13 tasks, benchmark 23 variations) as the target task family.

### Implemented
- Added `instant_policy/ip/generation_bimanual/`:
  - `config.py`: task list, benchmark variation counts, generation config.
  - `primitives.py`: 7 primitive families mapped to 13 RLBench2 bimanual tasks.
  - `generator.py`: emits `task_*.pt` files directly in `BimanualWorldBatch` format.
  - `README.md`: commands and contract.
- Added CLI:
  - `instant_policy/ip/scripts/generate_bimanual_pseudo_demos.py`
- Updated `instant_policy/ip/bimanual/dataset.py` collate path to support samples saved with explicit singleton batch dim `[1, ...]` (concatenate) and unbatched samples (stack).

### Primitive Set (locked)
1. `cooperative_lift`
2. `dual_push_sync`
3. `dual_push_transport`
4. `container_open_place_remove`
5. `handover`
6. `two_endpoint_tension`
7. `tool_plus_receptacle`

### Why
- This is the minimal reusable set that captures bimanual interaction structure without per-task ad-hoc code.
