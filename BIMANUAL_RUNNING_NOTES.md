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
3. **M2**: bimanual model heads and diffusion wrapper (relative targets only).
4. **M3**: dataset adapter with strict assertions on frame semantics.
5. **M4**: training entrypoint + smoke tests.

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
