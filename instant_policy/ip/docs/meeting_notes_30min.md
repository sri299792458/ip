# Instant Policy + Bimanual Update (30 min)

This is a presenter-ready slide outline with speaker script.
Use it as your meeting backbone and trim if time runs short.

---

## Slide 1 - Scope of Today's Update

### On-slide bullets
- Pseudo-demo generation: how it works and why these design choices.
- Single-arm training: paper anchors, current throughput, and speed knobs.
- Bimanual extension: representation contract + 7 primitive families.

### Speaker script
Today I will cover three things. First, how our pseudo demos are generated and why the generation design is physically meaningful while remaining scalable. Second, where we are on single-arm training relative to the paper runtime anchor and which knobs we are tuning for speed on MSI hardware. Third, how the bimanual branch is structured from first principles, including the invariance contract and our 7-primitive task decomposition.

---

## Slide 2 - Pseudo-Demo Pipeline (Single Arm)

### On-slide bullets
1. Build ShapeNet tabletop scene.
2. Sample object-centric waypoint specs.
3. Interpolate/resample trajectory at fixed motion spacing.
4. Simulate attach/detach from gripper state transitions.
5. Render 3-camera depth point clouds.
6. Package into training files (`data_*.pt` in active training mode).

### Speaker script
Our generator is fully kinematic and deterministic under seed control. We first sample a scene from ShapeNet meshes, then sample waypoint programs that are either structured skills or random object-centric behaviors. We convert these into smooth trajectories, resample at fixed translation and rotation increments, apply event-based grasp attach/detach, render object point clouds, and then serialize trainable step samples.

---

## Slide 3 - Pseudo-Demo Design Choices and Defaults

### On-slide bullets
- Object scale prior: `0.20..0.30 m`.
- Objects per scene (single-arm generator default): `2`.
- Waypoints per pseudo task: `2..6`.
- Skill mix: `50%` structured (`grasp/pick_place/pull/push`), `50%` random.
- Motion spacing: `1 cm`, `3 deg`.
- Disturbance probability: `0.3`.
- Gripper label-noise probability: `0.1`.
- Observation cameras: RLBench-style 3-camera rig, `128x128`.
- Training points per observation: `2048`.
- Debug video renderer: separate high-res visual renderer (does not change training observations).

### Speaker script
These defaults are selected to balance realism, diversity, and throughput. The 50/50 structured-vs-random split prevents overfitting to scripted patterns while still keeping strong manipulation priors. Camera intrinsics and layout follow RLBench-style assumptions. Importantly, video quality settings are decoupled from training observation settings, so visual debugging does not contaminate training data distribution.

---

## Slide 4 - Grasp/Attach Logic (First-Principles Version)

### On-slide bullets
- Gripper convention: `1=open`, `0=closed`.
- Attach check only on `open -> closed`.
- Detach on `closed -> open`.
- Front-only jaw-capture region (backside excluded).
- Threshold: `attach_capture_min_points=3`.
- No backside/full-mesh shortcut attach.

### Speaker script
The current rule is intentionally simple and physically interpretable. We only evaluate attach at the close event. The object must have enough surface points in the front jaw-capture zone in gripper-local coordinates. This avoids non-physical backside attachments and matches the intended semantics of closing around an object. We detach immediately on open.

---

## Slide 5 - Pseudo Debug Videos (How to Narrate)

### On-slide bullets
- `random`: broad stochastic behavior.
- `grasp`: approach -> close -> lift.
- `pick_place`: grasp -> move -> release.
- `pull`: contact + pulling trajectory.
- `push`: contact + pushing trajectory.
- What to verify: contact timing, motion continuity, and grip transitions.

### Speaker script
For each forced skill video, explain the expected sequence and verify three things: first, grip transitions occur at physically reasonable points; second, attached-object motion is coherent after close; third, trajectories are smooth and consistent with the sampled waypoint intent. This is our qualitative sanity check before large-scale runs.

---

## Slide 6 - Data Contract: File, Sample, Task, Batch

### On-slide bullets
- Active train path uses `steps` mode.
- One `data_*.pt` file = one trainable live-step sample.
- One pseudo task produces many `data_*.pt` files (not one).
- `BATCH_SIZE` = number of `data_*.pt` files per optimizer step.
- Valid live timestep requires future horizon availability (`t + H` exists).

### Speaker script
This is the key semantics slide to avoid confusion. In our active path, the atomic training item is a single step file. Pseudo tasks are expanded into many such files. So when we discuss ring size, batch size, and throughput, we are reasoning in step-files-per-second and optimizer-updates-per-second, not task-files-per-second.

---

## Slide 7 - Paper Anchors for Single-Arm Training

### On-slide bullets
- Context demos per sample: `N=2`.
- Context waypoints per demo: `L=10`.
- Prediction horizon: `T=8`.
- Training schedule: `2.5M` optimization steps + `50K` cooldown.
- Reported runtime anchor: ~`5 days` on one RTX 3080 Ti (with continuous generation/replacement).

### Speaker script
These values are our canonical reference points when discussing whether our pipeline is aligned. The key runtime anchor is based on optimizer steps, so update-rate (`it/s`) is the primary metric for walltime parity.

---

## Slide 8 - Runtime Math We Use

### On-slide bullets
- Required update rate for 5-day target:
  - `2.5e6 / (5*24*3600) ~= 5.79 it/s`
  - `2.55e6 / (5*24*3600) ~= 5.90 it/s` (with cooldown included)
- Consumption rate: `C = it/s * batch_size` (step-files/s).
- Generation rate: `P = tasks/s * files_per_task_avg` (step-files/s).
- Freshness proxy: `r = C / P` (`r~1` ideal-ish, `r>>1` generator lags).

### Speaker script
This is the throughput accounting we use. For paper-like walltime on the same step budget, we must sustain roughly 5.8 to 5.9 updates per second. Batch size changes sample throughput, but fixed-step walltime still hinges on update throughput.

---

## Slide 9 - Current Single-Arm Training Setup (Unified Script)

### On-slide bullets
- One-command path: `apptainer/train_instant_policy.slurm`.
- Core defaults:
  - `BATCH_SIZE=16`
  - `TRAIN_BUFFER_SIZE=8192`
  - `TRAIN_START_MIN_ITEMS=512`
  - `DEMOS_PER_TASK_MIN/MAX=2/4` (wrapper defaults)
  - `VAL_NUM_TASKS=100`
  - `TRAIN_NUM_WORKERS=8`, `TRAIN_PREFETCH_FACTOR=4`
  - `TRAIN_SAMPLE_CACHE_SIZE=2048`
  - `AUTO_RESUME=1`, `USE_WANDB=1`

### Speaker script
We consolidated into one slurm entrypoint to remove workflow fragmentation. Resume and logging are wired by default. The current defaults are intentionally conservative for stability, but they are the knobs we tune when targeting better throughput on H100.

---

## Slide 10 - What We Observed on MSI

### On-slide bullets
- A100 runs were around `~2.9 it/s` (below paper-parity target).
- H100 run reached about `~5.35 it/s` (close to target band).
- H100 run failed from host OOM kill (DataLoader-side memory pressure), not GPU compute collapse.

### Speaker script
The H100 test is important because it shows the model compute path is close to required update rate. The remaining issue is stability of the input pipeline under aggressive loader/cache settings. So the bottleneck shifted from pure GPU performance to host memory behavior in data loading.

---

## Slide 11 - Speed Knobs We Actively Tune

### On-slide bullets
- Compute/data pacing:
  - `BATCH_SIZE`
  - `TRAIN_NUM_WORKERS`
  - `TRAIN_PREFETCH_FACTOR`
  - `TRAIN_SAMPLE_CACHE_SIZE`
- Ring/generation behavior:
  - `TRAIN_BUFFER_SIZE`
  - `TRAIN_START_MIN_ITEMS`
  - `GEN_CHUNK_TASKS`
  - `DEMOS_PER_TASK_MIN/MAX`
- Storage/I/O:
  - `PCD_DTYPE` (`float16` active in slurm wrapper)

### Speaker script
These knobs are not equivalent. Some primarily affect update speed, some primarily affect memory, and some affect data freshness/diversity cadence. For example, `GEN_CHUNK_TASKS` changes burst granularity, while `TRAIN_BUFFER_SIZE` changes refresh window length, not direct update rate.

---

## Slide 12 - Bimanual Design Contract (First Principles)

### On-slide bullets
- Local observations:
  - `P^L = (T_W_L)^-1 P^W`
  - `P^R = (T_W_R)^-1 P^W`
- Explicit cross-arm relation:
  - `T_L_R = (T_W_L)^-1 T_W_R`
- Relative targets:
  - `DeltaT_L`, `DeltaT_R` (+ per-arm grips)
- Non-negotiable:
  - avoid broad raw world-frame pose channels as model shortcuts.

### Speaker script
The core bimanual decision is to keep the learning signal in relative/local terms that are invariant under global relabeling. Cross-arm coupling is explicit through `T_L_R`, so we preserve coordination without leaking brittle world-frame shortcuts.

---

## Slide 13 - Bimanual Pseudo-Data Scope and Semantics

### On-slide bullets
- Task family: 13 RLBench2 bimanual tasks.
- Variation counts in config sum to 23.
- Generation defaults:
  - `pred_horizon=8`
  - `min_steps=14`, `max_steps=24`
  - `num_points=2048`
  - `object_scale_range=(0.2,0.3)`
  - `num_objects_range=(3,5)`
  - `attach_capture_min_points=3`
- Grasp semantics:
  - attach on close transition only, detach on open,
  - targeted attach checks targeted object index,
  - contention: later successful close owns the object.

### Speaker script
This keeps bimanual pseudo generation consistent with single-arm philosophy while adding dual-arm semantics. The output samples are in `BimanualWorldBatch`-compatible format and feed the bimanual adapter that converts to relative/local representation.

---

## Slide 14 - Seven Primitive Families (Locked Set)

### On-slide bullets
1. `cooperative_lift`
2. `dual_push_sync`
3. `dual_push_transport`
4. `container_open_place_remove`
5. `handover`
6. `two_endpoint_tension`
7. `tool_plus_receptacle`

### Speaker script
We intentionally chose a minimal reusable primitive basis that covers the 13 RLBench2 bimanual tasks without per-task ad-hoc logic. This keeps the pseudo-data engine compositional and easier to scale/maintain.

---

## Slide 15 - Task-to-Primitive Mapping (Exact)

### On-slide bullets
- `bimanual_push_box` -> `dual_push_transport`
- `bimanual_lift_ball` -> `cooperative_lift`
- `bimanual_dual_push_buttons` -> `dual_push_sync`
- `bimanual_pick_plate` -> `cooperative_lift`
- `bimanual_put_item_in_drawer` -> `container_open_place_remove`
- `bimanual_put_bottle_in_fridge` -> `container_open_place_remove`
- `bimanual_handover_item` -> `handover`
- `bimanual_pick_laptop` -> `cooperative_lift`
- `bimanual_straighten_rope` -> `two_endpoint_tension`
- `bimanual_sweep_to_dustpan` -> `tool_plus_receptacle`
- `bimanual_lift_tray` -> `cooperative_lift`
- `bimanual_handover_item_easy` -> `handover`
- `bimanual_take_tray_out_of_oven` -> `container_open_place_remove`

### Speaker script
This mapping is encoded in code and is fixed for now. The point is to cover interaction motifs, not memorize individual tasks.

---

## Slide 16 - Bimanual Figure Walkthrough (What to Say)

### On-slide bullets
- Node types:
  - scene_left/right
  - gripper_left/right
- Typed relation edges:
  - scene-scene
  - scene-gripper
  - gripper-gripper
  - cross-arm gripper links
- Context panel -> current state transfer -> action rollout (`Action 1 ... Action T`).

### Speaker script
Use the figure to explain that the graph is typed and relation-aware, not a generic fully-connected blob. Emphasize explicit cross-arm coupling and context-to-current information flow before action denoising/rollout prediction.

---

## Slide 17 - Current Risks and Near-Term Plan

### On-slide bullets
- Single-arm runtime risk:
  - host RAM/DataLoader OOM under aggressive throughput settings.
- Bimanual status:
  - representation and pseudo-data scaffolding are in place.
- Immediate plan:
  - stabilize H100 runs with memory-safe loader/cache settings,
  - lock sustained `it/s` and estimate walltime for full step budget,
  - begin bimanual pseudo pretrain + RLBench2 fine-tuning sequence.

### Speaker script
The key blocker is now operational stability, not model definition. Once stable throughput is locked, we can scale single-arm confidently and execute the bimanual pretrain-to-finetune plan.

---

## Backup Slide A - Quick Q&A Answers

### Why local/relative frames for bimanual?
Because physically relevant relations remain stable under global rigid relabeling; world-frame shortcuts do not.

### Why 7 primitives only?
Minimal compositional basis with broad task-family coverage and lower maintenance burden.

### Why not interpret speed only with samples/sec?
Walltime parity against paper step budget depends on optimizer updates/sec (`it/s`), not only files/sec.

### Why OOM with moderate VRAM?
Dataloader workers/prefetch/LRU cache are host-RAM heavy; OOM can occur before VRAM saturates.

---

## Backup Slide B - Source Paths (for reproducibility)

- Single-arm generation docs: `instant_policy/ip/generation/README.md`
- Single-arm generation defaults: `instant_policy/ip/generation/config.py`
- Waypoint/skill sampler: `instant_policy/ip/generation/waypoint_sampler.py`
- Unified train script: `apptainer/train_instant_policy.slurm`
- Training flags/runtime: `instant_policy/ip/train.py`
- Paper constants quickref: `instant_policy/ip/paper.md`
- Bimanual notes: `BIMANUAL_RUNNING_NOTES.md`
- Bimanual generation docs: `instant_policy/ip/generation_bimanual/README.md`
- Bimanual primitives + mapping: `instant_policy/ip/generation_bimanual/primitives.py`
- Bimanual figure script: `instant_policy/ip/docs/make_bimanual_fig2.py`
