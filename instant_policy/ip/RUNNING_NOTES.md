# Deployment Running Notes

Last updated: 2026-02-11

## Log Discipline

### Decision
Keep this file as the canonical deployment decision log.

### Rule
- Any behavior change in `ip/deployment` gets a short note here in the same work session.
- Each note must include: what changed, why, and the user-visible effect.

## Single-Arm Train Defaults Fix (2026-02-11)

### Decision
Make online W&B logging the default for the unified single-arm SLURM pipeline and remove the misleading `999999` bootstrap default for pseudo generation.

### Changed
- `apptainer/train_instant_policy.slurm`
  - default `USE_WANDB` changed to `1`.
  - default `AUTO_RESUME` changed to `0` (resume is now explicit/opt-in).
  - train bootstrap default `TRAIN_NUM_TASKS` changed from `999999` to `TRAIN_BUFFER_SIZE`.
  - if ring buffer exists but is partially filled and `BOOTSTRAP_FILL_BUFFER=1`, bootstrap now auto top-ups with `--append --fill_buffer` until wrap.
  - top-up now runs only missing tasks (`TRAIN_BUFFER_SIZE - TRAIN_COUNT`) and sets `--task_start=TRAIN_COUNT` so progress and seeds continue from where the previous run stopped.
  - added startup logging for `use_wandb`, `train_buffer_size`, `train_num_tasks`, `fill_buffer`.
  - when `BOOTSTRAP_FILL_BUFFER=1` and `TRAIN_NUM_TASKS < TRAIN_BUFFER_SIZE`, auto-adjust bootstrap task budget to `TRAIN_BUFFER_SIZE`.
  - exports `WANDB_MODE=online` by default when W&B is enabled (unless caller already set `WANDB_MODE`).
- `ip/train.py`
  - default `--use_wandb` changed to `1`.
- `apptainer/run_instant_policy_vnc.sh`
  - auto-binds host `~/.netrc` to `/workspace/data/.netrc` when present.
  - ensures `/workspace/data/.netrc` permissions are tightened (`chmod 600`) at runtime.
- `apptainer/README_instant_policy.md`
  - updated defaults documentation for W&B and bootstrap task budget.

### Why
- Existing defaults silently disabled W&B online logging (`USE_WANDB=0`) even with valid credentials.
- `AUTO_RESUME=1` can hard-fail fresh runs when no checkpoint exists yet.
- With `--no-home`, host `~/.netrc` was not visible in-container unless explicitly bound.
- `TRAIN_NUM_TASKS=999999` was only a sentinel for fill-buffer early exit, but progress logs looked like an unbounded generation run.

### User-visible Effect
- `sbatch apptainer/train_instant_policy.slurm` now logs to W&B online by default.
- Bootstrap progress shows a bounded task budget by default instead of `999999`.

## Pseudo Render UV Shader Crash Fix (2026-02-11)

### Decision
Make pseudo-generation mesh rendering texture-agnostic by forcing plain-color visuals at mesh load time.

### Changed
- `ip/generation/scene_builder.py`
  - added `_force_plain_visual(...)` to convert loaded meshes to `ColorVisuals`.
  - call this conversion in `_load_mesh_base(...)` before normalization/scaling.

### Why
- Some ShapeNet assets include partial/broken texture metadata.
- Pyrender can compile a shader variant that references `uv_0` without the matching varying, causing a hard crash:
  `Shader compile failure ... undefined variable "uv_0"`.
- Pseudo training uses geometry only, so dropping texture fidelity is correct and safer.

### User-visible Effect
- Pseudo generation no longer dies mid-run from `uv_0` shader compile failures on problematic meshes.

## Single SLURM Pipeline Cleanup (2026-02-11)

### Decision
Keep exactly one SLURM entrypoint for pseudo-data + training and remove sweep-oriented helper scripts.

### Changed
- kept `apptainer/train_instant_policy.slurm` as the only SLURM flow (bootstrap pseudo train/val + train/resume).
- removed:
  - `apptainer/generate_pseudo_buffer.slurm`
  - `apptainer/generate_pseudo_val.slurm`
  - `ip/scripts/tune_attach_gates.py`
  - `ip/scripts/estimate_ring_buffer_plan.py`
- updated:
  - `apptainer/README_instant_policy.md` to document only the single-script workflow.
  - `ip/generation/README.md` to drop attach-gate sweep instructions.

### Why
- Reduce operational clutter and force one reproducible path on MSI.
- Remove tuning/sweep detours from the default workflow.

### User-visible Effect
- `sbatch apptainer/train_instant_policy.slurm` is now the single command path.
- No separate pseudo-generation or sweep scripts remain in the repo.

## Front-Only Capture Attach (2026-02-11)

### Decision
Replace mixed distance/capture attachment with a single first-principles rule:
front-only jaw-capture attach on close transition.

### Changed
- `ip/generation/pseudo_demo_generator.py`
  - jaw-capture region now clamps `z_min` to `0.0` in gripper-local policy frame (no backside capture).
  - `_should_attach` now ignores distance and attaches only when
    `cap_count >= attach_capture_min_points`.
- `ip/generation/config.py`
  - removed `attach_radius` from generation config.
- `ip/scripts/generate_pseudo_demos.py`
  - removed `--attach_radius` and its config wiring.
- `ip/scripts/tune_attach_gates.py`
  - removed `attach_radius` sweep axis; tuner now sweeps capture threshold only.
- `ip/generation/README.md`
  - updated defaults/flow/commands to reflect capture-only front-side attach.

### Why
- Full-mesh distance gating allows non-physical backside attaches.
- A single geometric capture rule is easier to reason about and aligns with the intended gripper behavior.

### User-visible Effect
- Pseudo grasps are now constrained to the front jaw region only.
- CLI/config surface is cleaner: no radius parameter for grasp attachment.

## Ring Buffer + Walltime Planner (2026-02-11)

### Decision
Add a dedicated planning utility for selecting trajectory ring-buffer size and estimating SLURM walltime before large training runs.

### Changed
- Added `ip/scripts/estimate_ring_buffer_plan.py`.
  - accepts measured train throughput (`train_steps_per_sec`) and pseudo generation throughput (`gen_tasks_per_sec_per_shard` or `gen_seconds_per_task_per_shard`).
  - evaluates multiple candidate ring sizes and reports:
    - per-shard task counts,
    - steps per epoch,
    - estimated fill/refresh time,
    - estimated pseudo demos in ring,
    - estimated disk usage (if task size is known).
  - estimates total training walltime for `num_iters` and recommends request time with safety factor.
  - optional task-dir probing to infer `avg_task_size_mb`, `avg_demos_per_task`, and `avg_frames_per_demo`.
  - optional JSON export for reproducible planning.
- Updated `apptainer/README_instant_policy.md` with planner usage examples.
- Updated `apptainer/generate_pseudo_buffer.slurm` and `apptainer/generate_pseudo_val.slurm`
  to expose `DEMOS_PER_TASK_MIN/MAX` as sbatch-exportable knobs.
- Updated `ip/train.py` + `apptainer/train_instant_policy.slurm` to support
  `num_iters_override` / `NUM_ITERS_OVERRIDE` for short throughput benchmark runs.

### Why
- Ring size and SLURM walltime choices were previously manual and easy to mis-size.
- Planning from measured rates gives a defensible request and avoids over/under-allocating MSI jobs.

### User-visible Effect
- You can now compute buffer-size tradeoffs and walltime requests from first principles in one command.

## Training Resume + W&B Robustness (2026-02-11)

### Decision
Make 24-hour-chunk training resumable with full trainer state, and clean up logging behavior.

### Changed
- `ip/train.py`
  - parses args once (removed repeated `parse_args()` calls).
  - initializes logger safely even when `record=1` and `use_wandb=0`.
  - added full-resume args:
    - `--resume_ckpt_path`
    - `--auto_resume` (latest checkpoint in `<save_path>/<run_name>`)
  - added W&B resume args:
    - `--wandb_id`
    - `--wandb_resume {allow,must,never}`
  - `trainer.fit(..., ckpt_path=...)` now uses resume checkpoint when requested.
- `apptainer/train_instant_policy.slurm`
  - added env knobs:
    - `AUTO_RESUME`
    - `RESUME_CKPT`
    - `WANDB_ID`
    - `WANDB_RESUME`
  - wires resume and wandb args through to `ip/train.py`.
- `apptainer/README_instant_policy.md`
  - added documented 24h resume workflow commands.

### Why
- MSI walltime cap requires continuing long runs across multiple jobs.
- Warm-start alone (`fine_tune`) does not restore trainer step/optimizer state.
- Logging needed to be robust in non-W&B runs.

### User-visible Effect
- You can continue the same training run across multiple 24h jobs with true checkpoint resume.
- W&B run continuation is supported when an existing run id is provided.

## Attach Defaults Update (2026-02-10)

### Decision
Set pseudo-generation default attach gate to the tuned setting from sweep results.

### Changed
- `ip/generation/config.py`
  - default `attach_radius` changed from `0.02` to `0.015`.
  - `attach_capture_min_points` kept at `3`.
- `ip/scripts/generate_pseudo_demos.py`
  - CLI default `--attach_radius` changed to `0.015`.
- `ip/generation/README.md`
  - paper/default section updated to reflect `attach_radius=0.015` and `attach_capture_min_points=3`.

### Why
- Sweep ranking favored the smaller radius with moderate capture threshold as the best recall/permissiveness tradeoff.

### User-visible Effect
- Running pseudo generation without explicit attach args now uses `attach_radius=0.015` and `attach_capture_min_points=3`.

## Attach-Gate Tuning Metrics Upgrade (2026-02-10)

### Decision
Upgrade attach-gate sweep metrics to penalize ambiguous/wrong-object attachment candidates, not only target-attach recall.

### Changed
- `ip/generation/pseudo_demo_generator.py`
  - added attach diagnostics:
    - `target_close_with_other_eligible`
    - `target_close_other_eligible_count_sum`
    - `hard_negative_other_eligible`
    - `hard_negative_other_eligible_count_sum`
  - hard-negative probe generation now supports object-scale offsets:
    - `offset = clip(scale * object_extent, min_m, max_m)`
  - retained optional fixed-offset probe mode.
- `ip/scripts/tune_attach_gates.py`
  - added new summary metrics:
    - `wrong_object_candidate_rate`
    - `hard_negative_other_eligible_rate`
    - mean eligible-count diagnostics
  - updated ranking score to include penalties for ambiguous/wrong-object eligibility.
  - added CLI args:
    - `--hard_negative_offset_scale`
    - `--hard_negative_offset_min_m`
    - `--hard_negative_offset_max_m`
    - optional override `--hard_negative_offset_m`
- `ip/generation/README.md`
  - updated tuning command to use object-scale hard-negative defaults.
  - documented fixed-offset override and new ranking intent.

### Why
- Previous tuning could overfit easy positives and report near-perfect recall without exposing permissive/wrong-object behavior.
- Object-scale probing makes hard negatives comparable across small and large meshes.

### User-visible Effect
- Sweep outputs are now more diagnostic and provide a clearer basis for selecting `attach_radius` and `attach_capture_min_points`.

## RLBench Runtime Camera Dump Utility (2026-02-10)

### Decision
Add a single script to dump RLBench camera intrinsics/extrinsics/near/far from the running simulator and verify scene-level vs reset-time values.

### Changed
- Added `ip/scripts/dump_rlbench_camera_info.py`.
  - Collects `Environment.get_scene_data()` camera payload.
  - Optionally resets a task and collects `obs.misc` camera payload.
  - Computes max-abs diffs between scene and reset snapshots.
  - Prints a concise summary and optionally writes JSON.

### Why
- Camera placement/scale/debug issues required exact runtime numbers from the actual simulator scene in Apptainer.
- This removes guesswork and gives a reproducible camera snapshot per run.

### User-visible Effect
- One command now produces machine-readable camera calibration details for all standard RLBench cameras.

## RLBench Workspace-Relative Camera Metrics (2026-02-10)

### Decision
Extend the camera dump utility with workspace/task-relative metrics to tune pseudo-demo camera placement against RLBench geometry.

### Changed
- `ip/scripts/dump_rlbench_camera_info.py`
  - records workspace center/bounds/size from simulator runtime.
  - records task base position after reset.
  - annotates per-camera offsets/distances to workspace center and task base.
  - includes these fields in JSON and terminal summary.

### Why
- Pseudo-demo camera tuning should match RLBench camera geometry relative to workspace/task, not world origin.

### User-visible Effect
- Camera dump JSON now directly provides the numbers needed to set pseudo camera poses from first principles.

## Pseudo Camera Rig Alignment to RLBench Geometry (2026-02-10)

### Decision
Update pseudo-demo default camera poses to match RLBench-style workspace-relative offsets from runtime camera dump.

### Changed
- `ip/generation/config.py`
  - `default_cameras()` now uses RLBench-derived offsets:
    - front: `[+1.10, 0.00, +0.828]`
    - left: `[-0.425, +0.20, +1.228]`
    - right: `[-0.425, -0.20, +1.228]`
  - offsets are anchored to pseudo target/workspace frame for stable composition.
- `ip/generation/README.md`
  - updated default-camera note to reflect RLBench-style rig.

### Why
- Camera placement changes depth-occlusion/coverage statistics even for point-cloud training.
- Aligning pseudo camera geometry to RLBench reduces train/eval distribution mismatch.

### User-visible Effect
- Generated pseudo point clouds now come from a more RLBench-like multi-camera viewpoint setup.

## Pseudo Camera Intrinsics/Resolution Alignment (2026-02-10)

### Decision
Align pseudo camera intrinsics and resolution to RLBench defaults, not just camera poses.

### Changed
- `ip/generation/config.py`
  - `default_cameras()` now sets RLBench-like intrinsics/resolution:
    - `width=height=128`
    - `fx=fy=175.839`, `cx=cy=64.0`
  - sets per-camera clipping planes from RLBench dump:
    - front `z_far=4.5`
    - left/right `z_far=3.2`
    - all `z_near=0.01`
- `ip/generation/README.md`
  - updated camera default note to include RLBench intrinsics/resolution alignment.

### Why
- Pose-only alignment left pseudo rendering on high-res intrinsics (`640x480`, `fx=525`), which increased per-task rendering/downsampling cost and mismatched RLBench observation geometry.

### User-visible Effect
- Pseudo generation is significantly lighter and camera statistics are closer to RLBench.

## Apptainer Display Process Cleanup (2026-02-10)

### Decision
Make container display-server lifecycle explicit to avoid lingering Xvfb/x11vnc/fluxbox processes and overlay shutdown timeouts.

### Changed
- `apptainer/run_instant_policy_vnc.sh`
  - `cleanup_stale_processes()` now kills all matching stale PIDs, not just the first match.
  - inside container shell, added `trap ... EXIT` cleanup for background `Xvfb`, `x11vnc`, and `fluxbox` processes.

### Why
- Repeated runs showed `INFO: Terminating fuse-overlayfs after timeout` caused by remaining background display processes after command completion.

### User-visible Effect
- Cleaner runner exits with fewer stale-process side effects across consecutive runs.

## Interpolation Stability Guardrails (2026-02-10)

### Decision
Harden trajectory interpolation against spherical-degeneracy cases and keep defaults on stable interpolation modes.

### Changed
- `ip/generation/trajectory_interpolator.py`
  - added robust fallbacks in spherical vector interpolation for near-opposite and ill-conditioned cases (`sin(theta) ~ 0`).
  - added finite-check fallback to linear interpolation when spherical output is invalid.
- `ip/generation/config.py`
  - default `interpolation_methods` changed from `("linear", "cubic", "spherical")` to `("linear", "cubic")`.
- `ip/generation/pseudo_demo_generator.py`
  - added fail-fast check for non-finite trajectory poses before rendering.

### Why
- Random-skill runs can produce interpolation corner cases that lead to non-finite poses and apparent hangs during rendering.

### User-visible Effect
- Random-skill by-task generation is more stable and avoids silent long stalls from invalid interpolation states.

## Video-Only Quality Increase (2026-02-10)

### Decision
Increase pseudo-debug video quality without affecting training observations.

### Changed
- `ip/generation/config.py`
  - added `render_visual_width=640` and `render_visual_height=640`.
- `ip/generation/renderer.py`
  - split renderers into:
    - observation renderer (uses camera config resolution, RLBench-aligned 128x128)
    - visual renderer (uses high-res output for debug videos)
  - scales intrinsics for visual rendering to preserve the same FOV.
- `ip/generation/pseudo_demo_generator.py`
  - wires visual render size config into `DepthRenderer`.
- `ip/generation/README.md`
  - documents that video quality is higher while training observations remain unchanged.

### Why
- Higher quality debug videos improve visual inspection, while training should keep RLBench-aligned observation statistics.

### User-visible Effect
- Rendered videos are sharper, but point-cloud observations used for training are unchanged.

## Pyrender Context-Binding Fix for Dual Renderers (2026-02-10)

### Decision
Use separate mesh caches for observation and visual offscreen renderers.

### Changed
- `ip/generation/renderer.py`
  - replaced single mesh cache with:
    - `mesh_cache_obs`
    - `mesh_cache_visual`
  - observation path uses obs cache; visual-video path uses visual cache.

### Why
- Pyrender mesh primitives are OpenGL-context bound.
- Reusing the same mesh primitive across two offscreen renderers raised:
  - `ValueError: Mesh is already bound to a context`

### User-visible Effect
- High-quality video rendering works with dual-renderer setup without crashing.

## Spherical Interpolation Re-Enabled for A/B (2026-02-10)

### Decision
Re-enable spherical interpolation in default method sampling to test whether prior long-stall behavior correlates with interpolation mode.

### Changed
- `ip/generation/config.py`
  - `interpolation_methods` set back to `("linear", "cubic", "spherical")`.
- Kept numerical guardrails in `trajectory_interpolator.py`.

### Why
- Needed a controlled check requested by user while avoiding silent non-finite pose failures.

### User-visible Effect
- Pseudo generation can sample spherical trajectories again for direct behavior comparison.

## Spherical Guardrails Toggle for Repro (2026-02-10)

### Decision
Add an explicit CLI switch to disable spherical interpolation guardrails for controlled reproduction testing.

### Changed
- `ip/scripts/generate_pseudo_demos.py`
  - added `--disable_spherical_guardrails` (debug-only).
- `ip/generation/config.py`
  - added `use_spherical_guardrails` config field (default `True`).
- `ip/generation/pseudo_demo_generator.py`
  - passes guardrail setting into trajectory interpolator.
- `ip/generation/trajectory_interpolator.py`
  - supports both guarded and legacy-unchecked spherical path math.
- `ip/generation/README.md`
  - documents debug flag.

### Why
- Needed to reproduce prior long-stall behavior without repeatedly editing source.

### User-visible Effect
- You can run safe default mode and legacy spherical mode with a single CLI switch.

## Spherical Repro Cleanup and Stable Revert (2026-02-10)

### Decision
After confirming spherical interpolation reproduces the long-stall behavior, revert to a clean stable baseline without debug toggles.

### Changed
- `ip/generation/config.py`
  - default interpolation methods reset to `("linear", "cubic")`.
  - removed temporary `use_spherical_guardrails` config field.
- `ip/generation/trajectory_interpolator.py`
  - removed legacy unchecked spherical branch used only for reproduction.
  - kept guardrailed spherical math implementation in code path (for explicit future use).
- `ip/scripts/generate_pseudo_demos.py`
  - removed temporary `--disable_spherical_guardrails` CLI flag.
- `ip/generation/pseudo_demo_generator.py`
  - removed wiring for temporary guardrail-toggle config.
- `ip/generation/README.md`
  - removed temporary debug-toggle documentation line.

### Why
- The repro objective was completed; keeping temporary toggles adds clutter and future confusion.

### User-visible Effect
- Default pseudo generation is back to the stable interpolation policy with cleaner CLI/config surface.

## Spherical Default Restored With Guardrails (2026-02-10)

### Decision
Keep `spherical` enabled in default interpolation method sampling, with guarded spherical math retained.

### Changed
- `ip/generation/config.py`
  - `interpolation_methods` set to `("linear", "cubic", "spherical")`.

### Why
- Reproduction showed unguarded spherical caused stalls, but guarded spherical is stable.
- User preference is to keep spherical behavior in default sampling once safety is confirmed.

### User-visible Effect
- Default pseudo generation samples spherical trajectories again, without reintroducing the previous unguarded stall mode.

## Spherical Support Consistency Fix (2026-02-10)

### Decision
Keep configuration and interpolator behavior consistent while retaining paper-aligned spherical option.

### Changed
- `ip/generation/trajectory_interpolator.py`
  - restored guarded `spherical` positional interpolation path.
  - method validation now accepts `linear`, `cubic`, `spherical`.

### Why
- A temporary intermediate edit removed spherical from interpolator while config still sampled it, which could trigger method-validation failures.

### User-visible Effect
- Default config including `spherical` now runs consistently again.

## Positional Spherical Removal (2026-02-10)

### Decision
Remove positional `spherical` interpolation from pseudo-demo trajectory generation and keep translation interpolation to `linear/cubic` only.

### Changed
- `ip/generation/trajectory_interpolator.py`
  - removed positional spherical branch and its helper vector-slerp path.
  - method validation now allows only `linear` and `cubic` for translation.
- `ip/generation/config.py`
  - default `interpolation_methods` set to `("linear", "cubic")`.
- `ip/generation/README.md`
  - clarified translation interpolation is linear/cubic; rotation interpolation remains quaternion Slerp.

### Why
- Two-point positional spherical interpolation is underconstrained; the midpoint-center formulation degenerates to near-antipodal vectors and mostly triggers fallback behavior.
- Keeping it implied “spherical” complexity without real geometric benefit.

### User-visible Effect
- Cleaner, deterministic translation interpolation behavior with no pseudo-spherical fallback ambiguity.

## Pseudo-Demo Category Debugging (2026-02-08)

### Decision
Add deterministic category forcing for pseudo-demo generation so each task type can be inspected independently in render videos.

### Changed
- `ip/scripts/generate_pseudo_demos.py`
  - added `--force_skill {auto,random,grasp,pick_place,pull,push}`.
  - `auto` keeps default stochastic behavior; other values force one category.
- `ip/generation/config.py`
  - added `forced_skill` to generation config.
- `ip/generation/waypoint_sampler.py`
  - supports deterministic category override when `forced_skill` is set.
  - validates allowed category names.
- `ip/generation/pseudo_demo_generator.py`
  - passes `forced_skill` into waypoint sampler.
- `ip/generation/README.md`
  - added loop command to render one debug video set per category.

### Why
- Random sampling makes it hard to isolate where a specific category is failing.
- Deterministic per-category runs make debugging penetration/attach/motion issues faster and repeatable.

### User-visible Effect
- You can now generate category-specific debug renders/videos without seed hunting.

## Pseudo-Skill Naming Cleanup (2026-02-08)

### Decision
Rename pseudo-task category labels from `open/close` to `pull/push` to avoid ambiguity with gripper open/close state.

### Changed
- `ip/generation/waypoint_sampler.py`
  - renamed category labels to `pull` and `push`.
  - renamed internal methods from `_open_waypoints/_close_waypoints` to `_pull_waypoints/_push_waypoints`.
- `ip/scripts/generate_pseudo_demos.py`
  - updated `--force_skill` choices to `auto,random,grasp,pick_place,pull,push`.
- `ip/generation/config.py`
  - updated `forced_skill` comment values.
- `ip/generation/README.md`
  - updated debug loop to use `pull/push` names.

### Why
- Prior naming suggested gripper commands rather than motion primitives.
- `pull/push` better matches the actual waypoint behaviors.

### User-visible Effect
- Debug/CLI category naming now reflects task semantics directly.

## Video-Only Render Export (2026-02-08)

### Decision
Allow pseudo-demo video generation without storing per-frame render PNGs.

### Changed
- `ip/generation/pseudo_demo_generator.py`
  - video writing now happens inline during trajectory rendering (streaming writer).
  - `--render_make_videos` no longer depends on `--save_renders`.
  - when `render_video_dir` is not set:
    - if `save_renders=true`, videos default under render folders.
    - if `save_renders=false`, videos default under `<save_dir>/_videos`.
- `ip/scripts/generate_pseudo_demos.py`
  - clarified CLI help text for `--save_renders`, `--render_make_videos`, and `--render_video_dir`.
- `ip/generation/README.md`
  - added explicit video-only command.
  - category-debug loop now renders videos without frame dumps.

### Why
- Frame PNGs consume significant storage and are unnecessary for routine debugging when MP4 output is sufficient.

### User-visible Effect
- You can run with `--render_make_videos` alone and only get videos on disk.

## Container Dependency Fix (2026-02-08)

### Decision
Make `trimesh` an explicit dependency of both local and apptainer environments.

### Changed
- Added `trimesh==4.11.0` to `apptainer/instant_policy.def` pip install list.
- Added `pyrender==0.1.45`, `pyopengl==3.1.0`, `pyglet==1.5.27`, `imageio`, `imageio-ffmpeg` to `apptainer/instant_policy.def`.
- Added `trimesh==4.11.0` to `instant_policy/environment.yml` pip section.
- `apptainer/run_instant_policy_vnc.sh` now auto-binds host `libGLU.so*` into the container if found.

### Why
- Pseudo-demo tooling (`ip.scripts.build_robotiq_mesh`) imports `trimesh`.
- Host `pip list` can show `trimesh`, but runtime inside container uses its own env.

### User-visible Effect
- Rebuilt container images include `trimesh` and stop failing with `ModuleNotFoundError: trimesh`.

## ShapeNet Index Path Robustness (2026-02-08)

### Decision
Make pseudo-demo generation robust when `--shapenet_index_path` parent directory does not exist.

### Changed
- `ip/generation/scene_builder.py` now creates the parent directory of `shapenet_index_path` before writing the index JSON.

### Why
- First-run generation failed with `FileNotFoundError` when `/workspace/data/pseudo_ring/` was missing.

### User-visible Effect
- Index creation no longer fails on missing parent folder; first run auto-creates it.

## Apptainer Path Defaults Cleanup (2026-02-08)

### Decision
Stop relying on `$HOME/ips` and manual shell exports for runner path resolution.

### Changed
- `apptainer/run_instant_policy_vnc.sh`
  - default `PROJECT_DIR` now resolves from script location (`repo_root = apptainer/..`).
  - `INSTANT_POLICY_DIR` keeps defaulting to `$PROJECT_DIR/instant_policy`.
  - canonicalizes `PROJECT_DIR`, `INSTANT_POLICY_DIR`, `DATA_DIR`, and `CONTAINER_IMAGE` to absolute paths.
  - upgraded shell strict mode to `set -euo pipefail`.
- `apptainer/README_instant_policy.md`
  - commands updated to `~/ip/apptainer`.
  - env var docs updated to reflect repo-relative default for `PROJECT_DIR`.
- `apptainer/build_instant_policy.sh`
  - canonicalizes script/repo/output/definition-file paths and uses absolute definition path for build.
  - supports `OUTPUT_DIR` override while keeping deterministic default behavior.

### Why
- Hardcoded home-folder naming is brittle across clones (`~/ips` vs `~/ip`).
- Relative-path/cwd assumptions in shell scripts are brittle and hard to debug.
- Repo-relative defaults plus canonical absolute paths remove per-shell setup friction.

### User-visible Effect
- Running `apptainer/run_instant_policy*.sh` works without exporting `PROJECT_DIR`/`INSTANT_POLICY_DIR` in typical repo layouts.

## Generation Docs Consolidation (2026-02-08)

### Decision
Keep one canonical markdown doc in `ip/generation` instead of split README + alignment note.

### Changed
- Rewrote `ip/generation/README.md` as the single clean generation reference.
- Removed `ip/generation/PAPER_ALIGNMENT.md`.
- Updated `ip/paper.md` mapping to point to `ip/generation/README.md`.

### Why
- Avoid duplicated/stale documentation and keep paper-alignment + operational usage in one place.

### User-visible Effect
- `ip/generation` now has one markdown entrypoint for all pseudo-demo generation details.

## Local Paper Corpus (2026-02-08)

### Decision
Keep a local full-text copy of the Instant Policy paper in repo for fast offline search.

### Changed
- Added:
  - `instant_policy/docs/papers/instant_policy/2411.12633v2.pdf`
  - `instant_policy/docs/papers/instant_policy/2411.12633v2.layout.txt`
  - `instant_policy/docs/papers/instant_policy/2411.12633v2.txt`
  - `instant_policy/docs/papers/instant_policy/SHA256SUMS`
  - `instant_policy/docs/papers/instant_policy/README.md`

### Why
- Avoid repeated online lookup when validating paper claims.
- `layout` and plain text variants improve search reliability across line-wrap styles.

### User-visible Effect
- Paper details can now be searched locally with `rg` at any time.

## Local Paper Quickref (2026-02-08)

### Decision
Keep a local paper quick-reference file in repo to avoid repeated online lookup for the same constants.

### Changed
- Added/filled `ip/paper.md` with:
  - canonical links (`arXiv`, `ar5iv`),
  - paper-canonical constants used in this codebase,
  - direct claim-to-code file mapping.

### Why
- Reduces repeated context switching and accidental drift when implementing or reviewing paper-aligned behavior.

### User-visible Effect
- Fast local reference for paper constants without needing a downloaded PDF.

## Training Default Alignment (2026-02-08)

### Decision
Make default training schedule paper-exact instead of open-ended.

### Changed
- `ip/configs/base_config.py`
  - `num_iters` set to `2,550,000` (2.5M train + 50K cooldown).
  - added `lr_cooldown_steps=50000`.
- `ip/models/diffusion.py`
  - added step LR schedule: constant LR until `num_iters - lr_cooldown_steps`,
    then linear decay to zero over the final `lr_cooldown_steps`.
  - this cooldown path now takes precedence over the optional cosine scheduler.

### Why
- Paper Appendix states training runs for 2.5M optimisation steps followed by a 50K LR cooldown period.

### User-visible Effect
- Default training now stops at paper-scale steps and applies a real cooldown by default.

## Ring-Buffer Training Pipeline (2026-02-08)

### Decision
Use trajectory-format ring buffer (`task_*.pt`) plus a fixed validation set, with SLURM scripts for generation and training.

### Changed
- Added `apptainer/generate_pseudo_buffer.slurm` for sharded pseudo-task ring generation.
- Added `apptainer/generate_pseudo_val.slurm` for fixed validation pseudo-task generation.
- Added `apptainer/train_instant_policy.slurm` for policy training with `--data_format trajectory`.
- Updated `apptainer/README_instant_policy.md` with end-to-end ring-buffer workflow and commands.

### Why
- Paper-style continuous pseudo-data generation is practical only with replacement.
- Trajectory storage avoids `data_*.pt` file explosion and keeps training I/O manageable.
- Fixed validation split gives stable model-selection signal while train buffer keeps refreshing.

### User-visible Effect
- You can now launch generation + training directly with:
  - `sbatch apptainer/generate_pseudo_buffer.slurm`
  - `sbatch apptainer/generate_pseudo_val.slurm`
  - `sbatch apptainer/train_instant_policy.slurm`

## Atomic Writes for Ring-Buffer Safety (2026-02-08)

### Decision
Write pseudo-data files atomically to avoid partial reads during concurrent generation/training.

### Changed
- `ip/utils/data_proc.py`: `save_sample` now writes `data_*.pt` via temp file + `os.replace`.
- `ip/generation/pseudo_demo_generator.py`: `task_*.pt` writes use temp file + `os.replace`.

### Why
- With continuous overwrite, readers can hit files while writers are updating.
- Atomic replace makes each file transition all-or-nothing.

### User-visible Effect
- Fewer silent data-loading failures when training runs alongside ring-buffer writers.

## Pseudo-Demo Gripper Mesh Source (2026-02-07)

### Decision
Use URDF-assembled Robotiq 2F-85 geometry for pseudo-demo generation.

### Changed
- Added `ip/scripts/build_robotiq_mesh.py`.
- Script supports:
  - auto-download of maintained `robotiq_description` files from `PickNikRobotics/ros2_robotiq_gripper`,
  - or using a local `robotiq_description` directory,
  - kinematic assembly from URDF joints/mimic chain at a selected jaw state,
  - output as a single `.obj/.ply/.stl` file.

### Why
- `--gripper_mesh_path` expects one mesh path while source geometry is multipart links.
- URDF assembly is the standard robotics way to compose link meshes and joint transforms.

### User-visible Effect
- One command now creates a URDF-faithful gripper mesh for:
  - `python -m ip.scripts.generate_pseudo_demos --gripper_mesh_path ...`

## Pseudo-Demo Paper-Fidelity Audit (2026-02-07)

### Decision
Align pseudo-demo generation strictly with Appendix D implementation details where claims are explicit.

### Changed
- `ip/generation/pseudo_demo_generator.py`
  - Removed proxy-gripper fallback; `gripper_mesh_path` is now required at runtime.
  - Attachment now triggers only on open->closed state transitions (detach on closed->open), matching the paper wording.
  - Removed synthetic point-cloud fallback on empty renders; generation now fails fast.
- `ip/scripts/generate_pseudo_demos.py`
  - `--gripper_mesh_path` is now required.
- `ip/generation/README.md`
  - Updated to state mesh path is required for paper-fidelity runs.

### Why
- Paper states pseudo-data is generated with an initialised Robotiq 2F-85 mesh and closest-object attach/detach on gripper state changes.
- Synthetic fallback point clouds are not consistent with "recorded using PyRender and simulated depth cameras."

### User-visible Effect
- Pseudo-demo generation now errors early if:
  - no Robotiq mesh path is provided, or
  - rendered observations are empty.

## Pseudo-Gen Dependency Fix (2026-02-07)

### Decision
Pin `pyrender` runtime dependencies in the environment spec.

### Changed
- Added to `environment.yml` pip section:
  - `pyrender==0.1.45`
  - `pyopengl==3.1.0`
  - `pyglet==1.5.27`

### Why
- `ip/generation/renderer.py` depends on `pyrender`, but these were not declared in the environment file.

### User-visible Effect
- Fresh `ip_env` setup now includes pseudo-demo rendering dependencies by default.
- Smoke test with `PYOPENGL_PLATFORM=egl` generated pseudo samples successfully.

## Pseudo-Gen README Sync (2026-02-07)

### Decision
Keep pseudo-demo README command snippets aligned with strict CLI requirements.

### Changed
- Updated `ip/generation/README.md` examples to include required `--gripper_mesh_path` in:
  - save-renders example,
  - steps ring-buffer example,
  - trajectory ring-buffer example.
- Updated troubleshooting note to reflect fail-fast behavior on empty rendered point clouds.

### Why
- CLI now requires a Robotiq mesh path and no longer falls back on synthetic point clouds.

### User-visible Effect
- README commands are copy-paste valid for current code.

## Robotiq Mesh State Convention (2026-02-07)

### Decision
Use a fixed 2F-85 mesh state (`open` by default) for proximity checks in pseudo-demo generation.

### Why
- Appendix D requires a Robotiq 2F-85 mesh and closest-object attach/detach on gripper state changes.
- It does not require articulated finger simulation during pseudo-demo synthesis.

### User-visible Effect
- `build_robotiq_mesh --state open` is the standard recommended artifact for `--gripper_mesh_path`.

## Gripper State Convention

### Decision
Use RLBench-style observed gripper state for `grips`:
- `grip = 1.0` (open) if measured openness `> 0.9`
- `grip = 0.0` (closed) otherwise

This applies to both:
- demo collection labels
- live deployment observations

### Why
- The model/data path in this repo is aligned with RLBench `gripper_open` semantics.
- Robotiq `OBJ` is contact/motion status, not open/closed aperture state.
- A single midpoint threshold like `0.5` on aperture is wrong for large-object grasps.

## What Was Removed

- OBJ-based gripper binarization from deployment/demo labels.
- `require_grip_objs` dependency in demo->model conversion and waypoint extraction.

## Current Operational Rule

- Gripper state is read from measured gripper position (`get_gripper_state`).
- We fail fast if gripper feedback is unavailable during collection/deployment.
- Waypoint extraction uses `grips` transitions from the final binary state.

## Reference Semantics

- Robotiq position convention: open near `0`, closed near `255`.
- In this repo's model convention: open `1`, closed `0` (inverted from normalized Robotiq position).
- RLBench observation uses open if `open_amount > 0.9`.

## Notes For Future Changes

If contact information is needed, keep it as a separate signal.
Do not overload `grips` with contact status.

## Viewer Defaults (2026-02-06)

### Decision
`ip.deployment.utils.view_demo_pcds` is now a minimal EE-frame viewer.

### Current Behavior
- Only input arg is `--demo`.
- Point clouds are always shown in end-effector frame (world -> EE transform).
- Playback controls are GUI-only: `Play`, `FPS`, `Frame`.
- Redundant `Frame idx` GUI readout removed.
- Viewer prints the actual bound URL using Viser runtime port.

### Why
- Keep the tool focused on one debugging task.
- Avoid confusion from stale fixed-port messages when Viser auto-selects another port.

## Debug Scope Cleanup (2026-02-06)

### Decision
Keep only three debug surfaces:
- Offline EE-frame PCD playback from `.pkl` (`view_demo_pcds`).
- Optional demo waypoint RGB+mask image export.
- Optional live outputs: `live.pkl` and per-step RGB+mask snapshots.

### Removed
- Real-time deployment viz toggles (`--viz`, `--viz-hz`).
- Old live policy-frame recorder (`--record-live-pcd` path).
- Always-on debug frame buffering during `capture_pcd_world`.
- `ip/deployment/debug_segmentation.py` (removed).
- `ip/deployment/debug_xmem_tracking.py` (removed).
- `ip/deployment/view_demo_live_alignment.py` (removed).

### Added / Changed
- `capture_pcd_world(..., capture_debug_frames=False)` is now opt-in for debug frame capture.
- Demo waypoint image export now uses RGB+mask sidecar frames (`_debug_frames`), not `_debug_rgb`.
- New deployment options:
  - `--save-live` to save live rollout as `ip/deployment/live.pkl`.
  - `--debug-live-frames` to save per-step RGB+mask images to `ip/deployment/debug_live_frames`.
- Live image overlays are minimal: `step`, `grip_raw`, `grip_bin`.

### Why
- Remove runtime/debug bloat from the main control loop.
- Keep debug artifacts optional and sidecar-only.
- Align debug outputs to what is needed to verify policy inputs.

## Docs Policy (2026-02-06)

### Decision
Temporarily remove `ip/deployment/README.md` and defer a clean rewrite until behavior stabilizes.

### Why
- Avoid maintaining stale docs while deployment interfaces are actively being simplified.

### Guide Discipline
- Keep `ip/deployment/docs/DEPLOYMENT_GUIDE.md` updated in the same change set whenever deployment behavior/CLI changes.
- Add new workflow sections when new functionality is introduced; do not only patch old snippets.

## Packaging + Entrypoints (2026-02-06)

### Decision
Remove deployment-time import/path hacks and make package entrypoints direct.

### Changed
- Added `ip/deployment/cli.py` as the canonical deployment CLI module.
- `ip/deployment/__main__.py` now imports `main` directly from `ip.deployment.cli`.
- Deployment debug scripts now import `build_default_config()` directly from `ip.deployment.cli`.

### XMem Import Rule
- Removed runtime `sys.path` mutation and repo-root path guessing from `xmem_segmentation.py`.
- XMem2 must be present on Python import path (documented via `.pth` registration in deployment guide).

### Why
- Avoid hidden path side effects and fragile `spec_from_file_location` behavior.
- Keep imports deterministic and packaging-friendly.
- Make setup failure explicit when XMem2 path registration is missing.

## Packaging Follow-up (2026-02-06)

### Decision
Remove legacy `setup.py` and use `pyproject.toml` as the single packaging source of truth.

### Validation
- `pip install -e . --no-deps` succeeds in `ip_env` with editable wheel build via `pyproject.toml`.

## Calibration Layout Cleanup (2026-02-06)

### Decision
Group calibration scripts under `ip/deployment/calibration/`.

### Moved
- `ip/deployment/calibrate_realsense_aruco.py` -> `ip/deployment/calibration/calibrate_realsense_aruco.py`
- `ip/deployment/compute_world_tag.py` -> `ip/deployment/calibration/compute_world_tag.py`
- `ip/deployment/validate_click_point.py` -> `ip/deployment/calibration/validate_click_point.py`
- `ip/deployment/view_frames_viser.py` -> `ip/deployment/calibration/view_frames_viser.py`

### Removed
- `ip/deployment/calibration/calibrate_realsense_relative.py`
- `ip/deployment/calibration/debug_dual_click_point.py`

### Behavior
- Script defaults target `ip/deployment/calibration/outputs/*.json` for calibration artifacts (`world_tag*.json`, `realsense_T_world_camera*.json`).
- Recommended invocation is now module-based (e.g. `python -m ip.deployment.calibration.compute_world_tag`).

### Viewer Simplification (Superseded)
- `ip/deployment/calibration/view_frames_viser.py` is now calibration-only:
  - defaults to all cameras in the calibration JSON (optional `--serial` filter),
  - removes axis-size tuning flags,
  - prints per-camera position and basic rotation sanity metrics (`det(R)`, orthogonality error).
- This was later removed; see "Calibration Viewer Removal (2026-02-06)" below.

## Deployment Layout Cleanup (2026-02-06)

### Decision
Move non-core loose files from `ip/deployment/` into typed subfolders.

### Moved
- Utility scripts to `ip/deployment/utils/`:
  - `inspect_demo.py`
  - `measure_workspace_bounds.py`
  - `replay_demo.py`
  - `set_home_position.py`
  - `view_demo_pcds.py`
- Manual XMem seeding helper to `ip/deployment/perception/manual_seed_xmem.py`.
- Markdown docs to `ip/deployment/docs/`:
  - `DEPLOYMENT_GUIDE.md`
  - `RUNNING_NOTES.md`
- Static JSON assets to `ip/deployment/assets/`:
  - `home_joint.json`

### Removed
- `ip/deployment/debug_demo.py` (redundant with current debug flow).
- `ip/deployment/show_demo_waypoints.py` (covered by `inspect_demo` + waypoint image export).

### Path Updates
- CLI default home path now points to `ip/deployment/assets/home_joint.json`.
- Utility command paths are now module-based under `ip.deployment.utils.*`.

## Naming + Control Layout (2026-02-06)

### Decision
Use `utils` naming for deployment helper scripts and place `RobotiqGripper` in `control/`.

### Changed
- Renamed deployment helper package:
  - `ip/deployment/tools/` -> `ip/deployment/utils/`
- Moved gripper driver:
  - `ip/deployment/ur/robotiq_gripper.py` -> `ip/deployment/control/robotiq_gripper.py`
- Removed stale `ip/deployment/ur/__init__.py`.

### Why
- `utils` better reflects helper-script intent.
- Gripper command interface belongs with motion/control stack.

## Calibration Outputs Consolidation (2026-02-06)

### Decision
Keep all calibration-generated matrices under `ip/deployment/calibration/outputs/`.

### Changed
- Moved `world_tag.json` and `world_tag_right.json` from `ip/deployment/assets/` to `ip/deployment/calibration/outputs/`.
- Removed legacy top-level `ip/deployment/calibration_outputs/`.
- Updated defaults/docs to reference `ip/deployment/calibration/outputs/...`.

### Why
- Calibration tools and their generated artifacts now live in one place.
- Avoids split-path confusion between static assets and calibration outputs.

## Calibration Viewer Removal (2026-02-06)

### Decision
Remove `ip.deployment.calibration.view_frames_viser` and keep `validate_click_point` as the single calibration sanity-check tool.

### Changed
- Deleted `ip/deployment/calibration/view_frames_viser.py`.
- Updated `ip/deployment/docs/DEPLOYMENT_GUIDE.md` to use `ip.deployment.calibration.validate_click_point` for calibration verification.

### Why
- Avoid duplicate calibration-debug paths.
- Keep calibration verification focused on the click-to-world check used in practice.

## Fail-Fast Cleanup (2026-02-06)

### Decision
Remove runtime fallbacks that silently continue with invalid/missing deployment data.

### Changed
- Calibration load is now strict in `ip/deployment/cli.py`:
  - missing `T_world_camera` or non-`4x4` matrix now raises immediately (no silent camera skip).
- Gripper feedback is now strict:
  - `URRTDEState.get_gripper_state()` no longer returns a default value.
  - Robotiq read methods now surface socket/protocol errors instead of returning `None`.
  - `URRTDEControl` now raises if gripper is enabled but unavailable.
- Segmentation/perception is now strict in `ip/deployment/perception/realsense_perception.py`:
  - with segmentation enabled, missing mask now raises (no fallback to unsegmented point clouds),
  - missing color/depth frames now raise per camera,
  - empty multi-camera capture now raises.
- XMem online segmenter now raises explicit errors for uninitialized cameras / failed seed / failed tracking instead of returning `None`.
- `ip/deployment/utils/replay_demo.py` now raises on invalid `T_w_e` frame shapes instead of skipping bad frames.
- `ip/deployment/utils/set_home_position.py` now raises on `--wait` timeout instead of warning and continuing.

### Why
- Avoid hidden degradation paths during demo collection and live deployment.
- Make sensor/segmentation/gripper failures explicit and actionable.

## Dependency Import Policy Cleanup (2026-02-06)

### Decision
Use direct imports for deployment dependencies instead of optional-import wrappers.

### Changed
- Removed optional import guards and `_require_*` helpers from:
  - `ip/deployment/cli.py`
  - `ip/deployment/control/ur_rtde_control.py`
  - `ip/deployment/state/ur_rtde_state.py`
  - `ip/deployment/utils/replay_demo.py`
  - `ip/deployment/utils/set_home_position.py`
  - `ip/deployment/utils/measure_workspace_bounds.py`
  - `ip/deployment/utils/view_demo_pcds.py`
  - `ip/deployment/perception/realsense_perception.py`
  - `ip/deployment/perception/manual_seed_xmem.py`
  - `ip/deployment/perception/sam_segmentation.py`
  - `ip/deployment/perception/xmem_segmentation.py`
  - `ip/deployment/calibration/calibrate_realsense_aruco.py`
  - `ip/deployment/calibration/validate_click_point.py`
  - `ip/deployment/orchestrator.py` (`cv2` debug import path)

### Exception Kept Intentionally
- `DemoCollector.collect_kinesthetic()` keeps runtime import for `pynput` hotkeys.
- Reason: in headless/no-X sessions, `pynput` import itself can fail due display backend, and this should only affect demo collection, not every deployment command.
- `ip/deployment/perception/xmem_segmentation.py` keeps a wrapped XMem2 import to provide a clear PYTHONPATH/`.pth` fix when XMem2 modules are missing.

### Why
- If the environment is missing required packages, fail at startup clearly instead of carrying optional wrappers across the codebase.

## CLI Surface Cleanup (2026-02-06)

### Decision
Trim non-essential CLI knobs and keep a smaller default interface for deployment/calibration utilities.

### Changed
- `ip.deployment` CLI:
  - Removed: `--debug-demo-waypoints-num`, `--manual-seed-out`, `--horizon-mode`,
    `--home-joints-deg`, `--home-joints-rad`, `--home-speed`, `--home-accel`,
    `--debug-gripper`, `--debug-frame-sanity`, `--debug-frame-every`.
  - Home move now always uses `ip/deployment/assets/home_joint.json` with fixed default move constants in code.
  - Execution horizon is fixed to `until-grip-change`.
- `ip.deployment.utils.replay_demo`:
  - Removed advanced motion/safety/gripper tuning flags; retained core replay controls only.
  - Replay uses fixed moveL defaults in code.
- `ip.deployment.utils.set_home_position`:
  - Removed `--joints-rad`, `--speed`, `--accel`, `--tolerance-deg`, `--max-wait-s`.
  - Keeps only essential degree/load/save/wait flow.
- `ip.deployment.calibration.validate_click_point`:
  - Removed robot motion-on-click options; tool is now calibration validation only (click pixel -> world point + optional TCP print).
- `ip.deployment.calibration.calibrate_realsense_aruco`:
  - Removed `--sleep-sec` and `--arm` convenience flag.
- `ip.deployment.calibration.compute_world_tag`:
  - Removed `--arm` convenience flag.
- `ip.deployment.utils.inspect_demo`:
  - Removed `--num-waypoints`; fixed to 10-waypoint inspection to match policy convention.

### Why
- Reduce CLI bloat and ambiguity.
- Keep defaults aligned with the standard deployment workflow already documented for this repo.

## Arm Selector Reintroduced (2026-02-06)

### Decision
Reintroduce a minimal `--arm {left,right}` only where calibration/default file naming depends on robot side.

### Changed
- `ip.deployment` now supports `--arm` and uses it to select default calibration JSON when `--calib` is not provided.
- `ip.deployment.calibration.compute_world_tag` now supports `--arm` to choose default output:
  - left: `world_tag.json`
  - right: `world_tag_right.json`
- `ip.deployment.calibration.calibrate_realsense_aruco` now supports `--arm` for:
  - default `--world-tag-matrix` fallback path (when explicit world-tag args are not provided),
  - default calibration output path (`realsense_T_world_camera*.json`).
- `ip.deployment.calibration.validate_click_point` now supports `--arm` to choose default calibration file.

### Why
- Left/right arm setups require different calibration artifacts.
- `--arm` avoids accidental cross-arm file reuse while keeping CLI surface small.

## CLI Trim Follow-up (2026-02-06)

### Decision
Remove path-override flags that add noise to the main deployment CLI.

### Changed
- Removed from `ip.deployment`:
  - `--home`
  - `--live-out`
  - `--debug-live-frames-dir`
- Fixed internal defaults:
  - home joints: `ip/deployment/assets/home_joint.json`
  - live rollout: `ip/deployment/live.pkl`
  - live frame debug dir: `ip/deployment/debug_live_frames`

### Why
- Keep the runtime surface minimal.
- Preserve advanced behavior only where it materially changes execution.

## CLI Trim Follow-up 2 (2026-02-06)

### Decision
Remove remaining deployment flags that create frame/path ambiguity.

### Changed
- Removed from `ip.deployment`:
  - `--debug-demo-waypoints-dir`
  - `--frame`
  - `--tcp-offset-m`
- Fixed internal default for demo-waypoint image export:
  - `ip/deployment/debug_waypoints`
- Deployment frame convention is now hard-fixed:
  - `FRAME = FLANGE`
  - `tcp_offset_in_code=False` in deployment CLI path.
- Demo task name is inferred from `--demo-out` filename stem (no extra CLI flag).

### Why
- Keep policy execution in one consistent frame.
- Avoid mixing deployment-frame choices with calibration offsets.
- Reduce path/config knobs that do not change core behavior.

## CLI Restore For Operator Workflow (2026-02-06)

### Decision
Restore a small set of deployment CLI overrides required by active collection workflow.

### Changed
- Restored on `ip.deployment`:
  - `--debug-demo-waypoints-dir`
  - `--debug-demo-waypoints-num`
  - `--frame`
  - `--tcp-offset-m`
- Current behavior:
  - default remains `--frame flange`,
  - `--frame tip --tcp-offset-m ...` is available when needed,
  - demo waypoint debug export can target custom dirs and waypoint count.
- `--task-name` stays removed; task label comes from `--demo-out` stem.

### Why
- Matches the exact command used during demo collection/debug in practice.
- Keeps defaults clean while preserving required expert overrides.

## CLI Restore Narrowed (2026-02-06)

### Decision
Keep only frame-control overrides restored; remove demo-waypoint path/count overrides again.

### Changed
- Kept restored on `ip.deployment`:
  - `--frame`
  - `--tcp-offset-m`
- Removed again from `ip.deployment`:
  - `--debug-demo-waypoints-dir`
  - `--debug-demo-waypoints-num`
- Waypoint debug export now always uses:
  - output dir: `ip/deployment/debug_waypoints`
  - waypoint count: `config.num_traj_wp`

### Why
- Preserve the required frame-control workflow.
- Avoid extra debug path/count knobs in the primary CLI.

## Frame Convention Simplification (2026-02-06)

### Decision
- Treat robot RTDE TCP as always flange in deployment workflow.
- Rename frame-related CLI flags to make the relationship explicit.

### Changed
- `ip.deployment`:
  - Replaced `--frame` with `--policy-ee-frame`.
  - Replaced `--tcp-offset-m` with `--flange-tip-offset-m`.
  - Startup now prints:
    - `ROBOT RTDE TCP FRAME = FLANGE (fixed)`
    - `POLICY EE FRAME = ...`
    - `FLANGE->TIP OFFSET (m) = ...` when tip mode is active.
- `ip.deployment.utils.replay_demo`:
  - Same CLI rename and startup print convention as `ip.deployment`.
- Frame metadata:
  - New demos now include `frame_spec` and `recorded_at_utc`.
  - Live rollouts (`live.pkl`) now include `frame_spec` and `recorded_at_utc`.
  - Deployment/replay validate demo `frame_spec` when present and fail on mismatch.

### Why
- Remove ambiguity between robot TCP definition and policy EE frame.
- Make frame assumptions explicit at command line and in saved artifacts.
- Prevent silent frame mismatch between collection, deployment, and replay.

## First-Principles Frame Convention (2026-02-06)

### Decision
- Robot RTDE TCP is always flange.
- Calibration uses `flange_to_calib_contact_m = 0.162` only in calibration tools.
- Demo/deployment/replay use one runtime offset: `flange_to_policy_origin_m` (default `0 0 0.088`).

### Changed
- Runtime CLI:
  - Removed `--policy-ee-frame`.
  - Removed `--flange-tip-offset-m`.
  - Added `--flange-to-policy-origin-m`.
- Runtime defaults:
  - `DeploymentConfig.tcp_offset_in_code=True`
  - `DeploymentConfig.tcp_offset_m=[0,0,0.088]`
- Frame metadata:
  - New schema stores `frame_spec.flange_to_policy_origin_m`.
  - Demo/replay validators support both new metadata and legacy (`policy_ee_frame` + `flange_tip_offset_m`) for compatibility.
- Calibration defaults remain unchanged:
  - `compute_world_tag --tcp-offset-m` default stays `0 0 0.162`.

### Why
- Removes unnecessary “tip vs flange mode” mental model from runtime.
- Separates calibration contact geometry from policy frame geometry.
- Keeps one consistent runtime convention across demo collection, deployment, and replay.

## World Tag Fit Upgrade (2026-02-06)

### Decision
Use four-corner best-fit for `compute_world_tag` instead of the older three-point construction.

### Changed
- `ip.deployment.calibration.compute_world_tag` now requires four corners:
  - `--tl`, `--tr`, `--br`, `--bl`
- Solver now:
  - fits a best-fit tag plane from all four points,
  - builds X/Y axes from averaged opposite edges,
  - reports plane-fit residuals (`rms`, `max`) as quality diagnostics.
- Offset flags were clarified:
  - `--flange-to-contact-m` (legacy alias: `--tcp-offset-m`)

### Input Rule
- Each corner input must be a 6D flange pose (`x y z rx ry rz`).
- 3D contact-point input mode was removed to keep calibration aligned with the fixed TCP=flange convention.

### Why
- Uses all available corner measurements, reducing sensitivity to per-corner touch noise.
- Removes ambiguity about when calibration offset should be active.

## Flange-Only Frame Metadata Enforcement (2026-02-06)

### Decision
Remove remaining non-flange legacy frame handling from deployment/replay metadata validation.

### Changed
- `ip.deployment.cli`:
  - removed legacy `policy_ee_frame` / `flange_tip_offset_m` compatibility path.
  - `frame_spec` is now required (missing metadata raises).
  - `frame_spec.flange_to_policy_origin_m` is now required (missing key raises).
- `ip.deployment.utils.replay_demo`:
  - same strict validation as `ip.deployment.cli`.
  - removed legacy `policy_ee_frame` / `flange_tip_offset_m` parsing.

### Why
- Keeps frame convention single-path and explicit: RTDE TCP is flange, plus one flange->policy-origin offset vector.
- Avoids silent or ambiguous behavior when replaying older frame metadata formats.

## Motion Control Alignment (2026-02-06)

### Decision
Align deployment motion defaults with UR/RTDE conventions and replace precomputed interpolation with a bounded feedback servo-step loop.

### Changed
- `ActionExecutor` now executes each policy target with a bounded closed-loop step:
  - each step uses current measured `T_w_e`,
  - bounds translation/rotation by safety limits per command,
  - repeats until target tolerance is reached (or hard substep cap).
- Updated defaults/constants:
  - `RTDEControlConfig.move_speed=0.25`, `move_acceleration=1.2`
  - `RTDEControlConfig.servo_time=0.002`, `servo_lookahead=0.1`, `servo_gain=300`
  - homing `moveJ` constants to `speed=1.05`, `acceleration=1.4`
  - replay `moveL` constants to `speed=0.25`, `acceleration=1.2`
- Removed custom deployment override that forced non-standard `moveL` defaults.

### Why
- Preserve Instant Policy target semantics (`T_target = T_base @ action_j`) without clipping policy outputs.
- Enforce safety at command execution level with feedback on real robot state.
- Keep defaults consistent with URScript/RTDE baseline conventions.

## Hardware Kinematic Guard + Strict RTDE Mode (2026-02-06)

### Decision
Add UR-side kinematic/safety prechecks in execution (not policy), and remove silent motion fallbacks.

### Changed
- `ActionExecutor`:
  - before each policy target, reads current joints and runs control-level target validation.
  - aborts early with explicit reason when target is not kinematically safe.
- `URRTDEState`:
  - added `get_actual_q()` with finite/shape checks.
- `URRTDEControl`:
  - added target validation path using:
    - `isPoseWithinSafetyLimits`
    - `getInverseKinematicsHasSolution(..., qNear=current_q)`
    - `getInverseKinematics(..., qNear=current_q)`
    - `isJointsWithinSafetyLimits`
  - `execute_pose()` now propagates explicit RTDE command failure.
  - invalid `control_mode` now raises immediately (no implicit fallback to `moveL`).
- `measure_workspace_bounds.py`:
  - removed stale `workspace_min/workspace_max` snippet fields (those fields are no longer part of `SafetyLimits`).

### Why
- Keeps singularity and kinematic handling in the hardware execution layer.
- Prevents silent runtime behavior when command mode is misconfigured.
- Avoids documenting non-existent safety fields.

## SPARK Pure-Python Stack Kickoff (2026-02-06)

### Decision
Implement a new SPARK teleop/data-collection runtime inside `ip/deployment` and use `SPARK-Remote-data_collection` only as reference.

### Changed
- Created new package: `ip.deployment.spark_teleop` with ROS-free foundations:
  - config model + JSON load/save
  - Spark serial packet reader and encoder unwrapping
  - UR runtime wrapper (RTDE + Robotiq)
  - teleop control loops (bimanual-capable)
  - Tk GUI monitor for Spark-vs-UR bounds
  - optional recorder + RealSense capture path
  - module entrypoint: `python -m ip.deployment.spark_teleop`
- Added guide: `ip/deployment/docs/SPARK_TELEOP_GUIDE.md`.

### Constraints Applied
- Parity-first with the existing SPARK behavior:
  - no ROS bus
  - no extra safety policy beyond SPARK-style enable-gated teleop path
  - GUI focused on teleop monitoring/controls rather than new visualization concepts.

## SPARK Teleop Parity + Fail-Fast Pass (2026-02-06)

### Decision
Tighten SPARK teleop runtime semantics to avoid silent misconfiguration and make freedrive behavior consistent with expected operator control.

### Changed
- `ip.deployment.spark_teleop.config`:
  - default config now auto-discovers local `offsets_lightning.pickle` / `offsets_thunder.pickle` when available.
  - if not found, keeps `offsets_pickle=""` (explicitly editable in config).
- `ip.deployment.spark_teleop.spark_serial`:
  - Spark packet parser now validates at least 7 encoder values.
  - if `offsets_pickle` is configured but missing, startup now raises (no silent fallback).
- `ip.deployment.spark_teleop.ur_runtime`:
  - entering freedrive stops active servo stream first.
  - `servo_j` exits freedrive before commanding.
  - added `is_freedrive_enabled()` for loop gating.
- `ip.deployment.spark_teleop.controller`:
  - command loop now pauses Spark servo while freedrive is active.
  - on command gating transitions (disable/stale/enable switch off), loop stops ongoing servo motion cleanly.
  - gripper command now keeps SPARK parity rounding (`0.1` steps after `[0,1]` clipping).

### Why
- Prevent hidden calibration-offset mismatches from bad file paths.
- Keep freedrive and Spark teleop modes behaviorally separated for operator clarity.
- Preserve fail-fast deployment principles while staying close to SPARK control flow.

## SPARK Unification Into Demo Collector (2026-02-06)

### Decision
Treat SPARK strictly as a demo control input backend and remove the parallel SPARK teleop stack.

### Changed
- Added Spark input driver in control layer:
  - `ip.deployment.control.spark_input`
  - includes Spark serial decode/unwrapping, profile mapping (`lightning`/`thunder`), and UR command loop.
- Added Spark alignment GUI in shared stack:
  - `ip.deployment.control.spark_alignment_gui`
  - runs throughout Spark collection and shows Spark-vs-UR XYZ error + bounds.
  - recording state is GUI-driven: `Start Recording`, `Stop + Save`, `Cancel`.
  - `Cancel` / window close discards the recording.
- Extended `URRTDEControl` with shared-control primitives used by Spark input:
  - `execute_joint_positions(...)` via `servoJ`
  - `set_gripper_closed_norm(...)`
  - `stop_motion()`
- Unified demo collection path:
  - `DemoCollector.collect_kinesthetic(...)` now supports `control_mode`:
    - `keyboard`: pendant freedrive + keyboard open/close hotkeys
    - `spark`: Spark serial teleop commands
  - both modes use identical camera/perception/state capture and demo output schema.
- CLI changes in `ip.deployment.cli`:
  - added `--demo-control {keyboard,spark}`
  - added Spark args for collect-demo mode:
    - `--spark-serial`
    - `--spark-profile {lightning,thunder}`
    - `--spark-offsets-pickle`
- Removed redundant package `ip/deployment/spark_teleop/` (camera/runtime/gui/recorder parallel stack).
- Rewrote `ip/deployment/docs/SPARK_TELEOP_GUIDE.md` to document Spark-as-input workflow on the shared pipeline.

### Why
- Avoid duplicate camera and runtime implementations.
- Ensure keyboard and Spark collection produce the same data path and semantics.
- Keep SPARK scope minimal: substitute for freedrive+keyboard input only.

## SPARK Flow Audit + Start/Stop Consistency (2026-02-06)

### Decision
Keep Spark as control-input-only in the shared collector path, with GUI alive throughout collection and explicit record start/stop semantics.

### Changed
- Audited `SPARK-Remote-data_collection` behavior:
  - teleop GUI runs continuously
  - recording start/stop is a separate operator trigger
- Confirmed unified path in deployment:
  - one collector pipeline for camera/state/demo schema
  - Spark only substitutes the command source
- Fixed Spark pre-record wait behavior:
  - wired keyboard `q/esc` stop event into GUI `wait_for_start(...)` external stop callback.
  - this ensures clean cancel before start without hanging in pre-record wait.

### Why
- Matches original Spark operator mental model (monitoring GUI always on, explicit recording control).
- Preserves identical data semantics between keyboard and Spark collection modes.
- Removes edge-case mismatch in cancel flow before recording starts.

## Waypoint Selection Refinement (2026-02-06)

### Decision
Refine fixed-10 waypoint extraction to prioritize task events and geometric coverage, while removing pause redundancy.

### Changed
- Updated `ip/utils/data_proc.py::extract_waypoints(...)`:
  - keeps strict binary gripper-state requirement (`{0,1}`) for event semantics.
  - adds a motion-compression pass before waypoint selection:
    - drops near-static frames unless a gripper transition occurs.
  - keeps mandatory anchors:
    - start, end, gripper transition frame, and pre-transition frame.
  - fills remaining slots to `num_waypoints` using arc-length coverage over compressed trajectory segments.
  - fallback fills any shortfall using farthest-by-arc points; pads with final frame only when unavoidable.

### Why
- Reduces wasted waypoint slots on long pauses / jitter.
- Keeps open/close stage boundaries explicit for conditioning.
- Makes selection less sensitive to recording speed (keyboard vs Spark) by using geometric progress instead of time.

## Waypoint Selector Simplification (2026-02-06)

### Decision
Keep the same first-principles behavior, but reduce implementation complexity.

### Changed
- Simplified `ip/utils/data_proc.py::extract_waypoints(...)` fill stage:
  - removed per-segment budget allocation and interpolation-placement logic.
  - now uses one deterministic rule after anchors:
    - farthest-point sampling over cumulative SE(3) arc length.
- Kept unchanged:
  - motion compression pass,
  - mandatory grip-transition anchors,
  - strict binary grip-state check.

### Why
- Easier to reason about and maintain.
- Preserves the intended behavior (event preservation + geometric coverage + pause suppression) without extra machinery.

## Running Notes Relocation + Language Doc Refresh (2026-02-07)

### Decision
Move running notes to top-level `ip/` and keep the language modality transfer document concise and code-aligned.

### Changed
- Moved notes file:
  - `ip/deployment/docs/RUNNING_NOTES.md` -> `ip/RUNNING_NOTES.md`
- Rewrote `ip/README_LANGUAGE_MODALITY_TRANSFER.md` to match current implementation and remove outdated narrative bulk.

### Why
- Notes are cross-cutting (deployment, training, data), so top-level placement is easier for ongoing updates.
- The previous language document was hard to scan and had stale sections that were no longer useful for implementation decisions.

## Language CLI Help Alignment (2026-02-07)

### Decision
Align `train_language.py` and `eval_language.py` CLI help text with the streamlined language-modality-transfer guide.

### Changed
- Updated `argparse` descriptions/help for:
  - `ip/train_language.py`
  - `ip/eval_language.py`
- Clarified:
  - checkpoint directory expectations (`model.pt` + `config.pkl`)
  - dataset expectation (`data_*.pt` with `lang_emb`)
  - language input modes at eval (`--lang_emb_path`, `--lang_text`, `--paraphrase_file`)
- Set `eval_language.py` default `--model_path` to `./checkpoints/ip` to match current guide and repository layout.
- Moved RLBench-dependent imports in `eval_language.py` to lazy import sites so `python -m ip.eval_language --help` works even when RLBench is not installed.

### Why
- Reduces command ambiguity and mismatch between docs and executable CLI behavior.
- Makes common usage discoverable directly from `--help`.

## Apptainer Popup Fix + Non-VNC Runner (2026-02-07)

### Decision
Fix CoppeliaSim update-popup suppression using the actual settings keys read by CoppeliaSim, and add a no-VNC runner for training jobs.

### Changed
- Updated `apptainer/run_instant_policy_vnc.sh`:
  - writes `doNotShowUpdateCheckMessage=1`, `suppressStartupDialogs=1`, `noVersionCheck=1` into `usrset.txt`,
  - removes legacy `checkForUpdates*` keys from `usrset.txt`,
  - unsets `XDG_CONFIG_HOME` to make settings path deterministic (`$HOME/.CoppeliaSim/usrset.txt`),
  - supports `RLBENCH_ENABLE_VNC=0` (Xvfb-only mode).
- Added `apptainer/run_instant_policy.sh` wrapper that sets `RLBENCH_ENABLE_VNC=0` and reuses the same runtime setup.
- Updated `apptainer/train_language.slurm` default runner to `run_instant_policy.sh`.
- Updated `apptainer/README_instant_policy.md` to document the no-VNC mode and corrected popup-suppression behavior.

### Why
- The previous keys were not consumed by CoppeliaSim, so the popup could still block evaluations.
- Training does not need interactive VNC by default; Xvfb-only is cleaner and lighter while preserving compatibility.

## Apptainer Runner Defaults Cleanup (2026-02-07)

### Decision
Use no-VNC runner by default for non-interactive batch workflows; keep VNC runner only for workflows that actually require interactive display streaming.

### Changed
- Updated default runner in:
  - `apptainer/generate_rlbench_data.slurm`
  - `apptainer/convert_peract.slurm`
  - both now default to `run_instant_policy.sh` (Xvfb-only).
- Removed stale backup file:
  - `apptainer/run_instant_policy_vnc.sh.backup`

### Why
- `generate_rlbench_data` already runs in `--headless` mode.
- `convert_peract` is offline dataset conversion.
- Keeping VNC off by default reduces process overhead and avoids unnecessary ports/processes.

## Pseudo-Demo First-Principles Alignment (2026-02-07)

### Decision
Align pseudo-demonstration generation with the paper and with the repo-wide RLBench gripper convention.

### Changed
- `ip/generation/waypoint_sampler.py` now uses explicit convention constants:
  - `OPEN = 1`, `CLOSED = 0`
  - biased skill waypoint templates updated accordingly.
- `ip/generation/pseudo_demo_generator.py` attachment logic now follows:
  - attach on open->closed transition using closest-object selection,
  - detach on closed->open transition.
- `ip/generation/augmentation.py` gripper corruption now matches Appendix D:
  - `gripper_noise_prob=0.1` is applied per timestep (not per trajectory).
- `ip/generation/config.py` default clarified:
  - `gripper_noise_prob` comment updated to per-timestep probability,
  - `randomize_num_demos` default set to `False` (fixed context count unless explicitly enabled).
- `ip/generation/__init__.py` switched to lazy export of `PseudoDemoGenerator` so utility imports do not require `pyrender`.
- `ip/generation/README.md` updated:
  - fixed gripper convention statement,
  - corrected augmentation semantics,
  - removed stale `render_to_video.py` reference,
  - updated examples for current scripts and MSI-style ShapeNet path.

### Why
- Training/eval/deployment in this repo use RLBench gripper semantics (`1=open`, `0=closed`).
- Previous pseudo-demo defaults mixed opposite semantics and trajectory-level grip noise, which drifted from the paper and introduced avoidable train/eval mismatch.

## Pseudo-Demo Debug Script Consolidation (2026-02-07)

### Decision
Remove ad-hoc pseudo-demo helper scripts and keep one minimal debug utility.

### Changed
- Deleted:
  - `ip/scripts/plot_pseudo_sample.py`
  - `ip/scripts/animate_pseudo_demo.py`
  - `ip/scripts/merge_pseudo_demos.py`
- Added:
  - `ip/scripts/debug_pseudo_demo.py`
  - single-frame PNG debug from `data_*.pt`
  - multi-frame GIF debug from `data_*.pt`
- Updated `ip/generation/README.md` examples to use only `debug_pseudo_demo.py`.

### Why
- Keep one clear, maintained path for pseudo-demo visual sanity checks.
- Remove overlapping scripts with partially duplicated logic and stale options.

## Pseudo-Demo Paper Fidelity Pass (2026-02-07)

### Decision
Bring pseudo-demo generation closer to Appendix D wording by explicitly modeling a Robotiq 2F-85 mesh and preserving pseudo-task consistency across demos.

### Changed
- `ip/generation/pseudo_demo_generator.py`:
  - initializes a gripper mesh for pseudo-demo synthesis:
    - uses `--gripper_mesh_path` when provided, otherwise a metric 2F-85 proxy mesh,
  - uses gripper-mesh surface distance (not only gripper-origin distance) to find closest object on close transitions,
  - keeps optional `attach_radius` threshold, with default `None` (closest-on-close behavior),
  - removed per-demo waypoint perturbation to keep sampled waypoint specs semantically consistent across demos of one pseudo-task.
- `ip/generation/renderer.py` now accepts external gripper mesh for visual rendering.
- `ip/scripts/generate_pseudo_demos.py` adds `--gripper_mesh_path`.
- `ip/generation/config.py`:
  - adds `gripper_mesh_path`,
  - sets `attach_radius=None` default.

### Why
- Appendix D explicitly states pseudo-demo generation initializes a Robotiq 2F-85 gripper mesh and attaches/detaches closest objects on gripper-state changes.
- Waypoint perturbation per demo weakened pseudo-task consistency; object pose randomization and start-pose randomization are sufficient and closer to the paper description.

## Pseudo-Demo Penetration Fix + Scale Prior (2026-02-08)

### Decision
Keep object size prior at `0.07..0.13 m` and fix gripper-object penetration via explicit contact gating and pose de-penetration.

### Changed
- `ip/generation/config.py`:
  - `object_scale_range` default set to `(0.07, 0.13)`,
  - `attach_radius` default set to `0.02` (meters),
  - added `depenetration_clearance_m=0.003`,
  - added `depenetration_max_iters=4`.
- `ip/scripts/generate_pseudo_demos.py`:
  - CLI default `--object_scale_range 0.07 0.13`,
  - added `--attach_radius`,
  - added `--depenetration_clearance_m`,
  - added `--depenetration_max_iters`.
- `ip/generation/pseudo_demo_generator.py`:
  - closest-object query now returns closest distance and push direction,
  - added per-step de-penetration projection before rendering/state save,
  - attach now requires contact (`closest_dist <= attach_radius`) on open->closed transition,
  - rendered/saved `T_w_e` now use the corrected (de-penetrated) pose.
- `ip/generation/README.md`:
  - documented `0.07..0.13 m` prior and the new attach/de-penetration behavior.

### Why
- Unbounded nearest-object attach (no distance gate) can create unrealistic teleport-style grasping.
- Pose noise/interpolation can place the gripper mesh slightly inside objects; projecting to a small positive clearance removes this artifact while keeping trajectories smooth.
- Keeping corrected poses in saved demos avoids observation/pose mismatch.

## Pseudo-Demo Depenetration Direction Fix (2026-02-08)

### Decision
Use signed-distance directionality for depenetration to avoid pushing deeper into objects when a sampled gripper point is already inside geometry.

### Changed
- `ip/generation/pseudo_demo_generator.py`:
  - closest-object query now uses mesh nearest + signed distance in object frame,
  - inside-object points (`signed_distance > 0`) invert push direction (outward),
  - per-step depenetration now selects the most violating object based on:
    - penetration depth (if any), else
    - clearance deficit to nearest surface.

### Why
- Unsigned nearest-distance alone cannot distinguish inside vs outside and can produce wrong push direction under penetration.
- Signed-distance-based correction directly targets real intersection and prevents the “gripper sits inside object” failure mode.

## Pseudo-Demo First-Principles Collision Projection (2026-02-08)

### Decision
Replace per-step push-out heuristics with collision-feasible trajectory projection along the commanded motion segment.

### Changed
- `ip/generation/pseudo_demo_generator.py`:
  - added signed-distance clearance metric (`_max_clearance_violation`),
  - start pose sampling now prefers collision-free initial poses,
  - replaced push-based depenetration with:
    - SE(3) interpolation (`_interpolate_pose`),
    - bisection line-search projection (`_project_pose_from_prev`) that finds the largest feasible `alpha` on `prev -> desired`,
  - render loop now tracks `prev_pose` and projects each step before attach/render/save.
  - this supersedes the prior local push-out depenetration path.

### Why
- This enforces a clean invariant: non-attached objects are not penetrated by accepted gripper poses.
- It avoids direction/overshoot artifacts from local push vectors and is easier to reason about as a deterministic feasibility projection.

## Pseudo-Demo Simplification: Frame Alignment over Collision Projection (2026-02-08)

### Decision
Return to a simpler paper-style generation loop (waypoints + attach/detach) and fix severe penetration at the source via gripper frame alignment.

### Changed
- `ip/generation/pseudo_demo_generator.py`:
  - removed collision-projection/depenetration path,
  - kept simple event-based attach/detach with `attach_radius` gate,
  - added gripper mesh frame canonicalization:
    - translate loaded Robotiq mesh from URDF/base-link frame to policy-origin frame (`z=0.088 m`).
- `ip/generation/config.py` and `ip/scripts/generate_pseudo_demos.py`:
  - removed depenetration config/CLI fields.
- `ip/generation/README.md`:
  - updated to reflect simplified flow and policy-origin frame canonicalization.

### Why
- Paper Appendix D uses a simple kinematic process and explicitly does not enforce full kinematic feasibility.
- Severe penetration observed in our outputs was primarily a frame-convention mismatch (URDF base-frame mesh vs grasp-centric waypoint sampling), not a need for a full collision solver.
- Frame canonicalization preserves simple logic while removing the worst failure mode.

### Convention Check
- This aligns pseudo generation with deployment convention:
  - robot flange/base frame -> policy origin offset is `0.088 m` along tool `+Z`.
- Model gripper node template remains unchanged; only mesh used for attach/render proximity is re-referenced.

## Pseudo-Demo Simplification: Remove `rtree` Requirement (2026-02-08)

### Decision
Keep closest-object matching simple and dependency-light by using sampled surface-point distances only.

### Changed
- `ip/generation/pseudo_demo_generator.py`:
  - `_closest_object` no longer calls `trimesh.proximity.closest_point`,
  - uses pairwise distance between transformed object sampled points and gripper sampled points.
- Removed `rtree==1.3.0` from:
  - `apptainer/instant_policy.def`
  - `instant_policy/environment.yml`

### Why
- This restores the simple kinematic pseudo-data path without requiring spatial-index dependencies.
- Avoids environment fragility across nodes while preserving the intended closest-object attach behavior.

## Pseudo-Demo Rigid Attach Fix (2026-02-08)

### Decision
Decouple gripper-label noise from attachment dynamics so grasped objects remain rigidly attached during simulated motion.

### Changed
- `ip/generation/augmentation.py`:
  - added `augment_motion(...)` for pose-only augmentation,
  - added `augment_gripper_labels(...)` for label-only grip corruption.
- `ip/generation/pseudo_demo_generator.py`:
  - render/attach simulation now uses motion-augmented trajectory with clean grip states,
  - 10% gripper-state noise is applied only after rendering to saved `grips`.
- `ip/generation/README.md`:
  - documented that gripper noise is post-simulation label corruption.

### Why
- Previously, per-timestep grip flips were injected before attachment simulation, causing artificial detach/reattach jitter and non-rigid “sliding” while grasped.
- Paper-style robustness noise can be preserved without corrupting rigid-attachment kinematics.

## Pseudo-Demo Grasp Sampling Robustness (2026-02-08)

### Decision
Fix pass-through grasp starts via better object-centric waypoint sampling, without adding collision solvers.

### Changed
- `ip/generation/waypoint_sampler.py`:
  - top-down grasps now sample only top-facing surface points (`normal_z >= 0.2`) when available,
  - surface normal orientation is corrected using center->surface radial direction (robust to mesh winding issues),
  - grasp offset lower bound changed from `0.0` to `0.01 m` (non-zero clearance).

### Why
- Top-down approach on arbitrary surface points (including underside/side) can produce paths that appear to go through the object before grasp.
- Zero offset can place the gripper immediately at contact/intersection.
- This remains simple and paper-faithful (no explicit collision feasibility solver).

## Pseudo-Demo Object-Centric Attach Semantics (2026-02-08)

### Decision
Make close-event attachment explicitly object-centric when waypoint specs target a specific object.

### Changed
- `ip/generation/waypoint_sampler.py`:
  - `Waypoint` now carries `obj_index`,
  - `resolve_waypoints` preserves `obj_index` from sampled specs.
- `ip/generation/trajectory_interpolator.py`:
  - interpolated waypoints preserve `obj_index` per segment endpoint.
- `ip/generation/augmentation.py`:
  - trajectory copy preserves `obj_index`.
- `ip/generation/pseudo_demo_generator.py`:
  - on open->closed, if current waypoint has `obj_index`, attachment is attempted only for that object (with `attach_radius` gate),
  - nearest-object fallback is used only for waypoints without object target.

### Why
- First-principles object-centric generation should not randomly attach a different object during intended grasp transitions.
- This removes a major source of non-rigid/semantically inconsistent demos while keeping the generator simple and kinematic.

## Pseudo-Demo Render Styling (2026-02-08)

### Decision
Improve debug-render readability with white background and blue gripper.

### Changed
- `ip/generation/renderer.py`:
  - render scenes use `bg_color=[1,1,1,1]`,
  - added blue material for visual gripper mesh in `render_visual`.

### Why
- Easier visual inspection of gripper/object interaction in exported debug frames/videos.

## Pseudo-Demo Scale Prior Update (2026-02-10)

### Decision
Set the default pseudo object metric scale prior to `0.20..0.30 m` for current experiments.

### Changed
- `ip/generation/config.py`:
  - `object_scale_range` default updated from `(0.07, 0.13)` to `(0.2, 0.3)`.
- `ip/scripts/generate_pseudo_demos.py`:
  - CLI default `--object_scale_range` updated from `0.07 0.13` to `0.2 0.3`.
- `ip/generation/README.md`:
  - default scale prior documentation updated to `0.20..0.30 m`.

### Why
- Current pseudo renders looked gripper-dominant with smaller object prior.
- This establishes a larger object-size prior as the repo default so manual CLI overrides are no longer required each run.

## Pseudo-Demo Attach Logic Upgrade (2026-02-10)

### Decision
Replace distance-only attach gating with a combined rule:
- distance-to-gripper surface (existing `attach_radius`), and
- jaw-capture region occupancy (for thin objects between open fingers).

### Changed
- `ip/generation/pseudo_demo_generator.py`:
  - added mesh-derived jaw-capture region estimation in gripper frame,
  - added object capture-count check at close transitions,
  - attach now succeeds when either distance gate passes or capture-count gate passes,
  - preserved object-centric behavior: if waypoint targets object `k`, only `k` is eligible on close,
  - for non-object-centric waypoints, choose best attach candidate by capture support then distance.
- `ip/generation/README.md`:
  - updated attach semantics documentation to match the new gating rule.

### Why
- Distance to an *open* gripper mesh can miss valid grasp situations for thin geometries (e.g., table legs) that lie between fingers before close.
- The jaw-capture gate better matches rigid close-event attachment while staying fully kinematic and paper-style (no collision solver).

## Attach Tuning Sweep Tool (2026-02-10)

### Decision
Add a deterministic sweep script to tune attach thresholds from metrics, not from ad-hoc visual inspection.

### Changed
- `ip/generation/config.py`:
  - added `attach_capture_min_points` (default `3`) as a first-class config parameter.
- `ip/scripts/generate_pseudo_demos.py`:
  - added CLI argument `--attach_capture_min_points`.
- `ip/generation/pseudo_demo_generator.py`:
  - replaced hardcoded capture threshold with config-driven parameter,
  - added no-render simulation path and attach event stats collection,
  - added `evaluate_demo_attach_stats(...)` and `evaluate_task_attach_stats(...)` for fast metric sweeps.
- `ip/scripts/tune_attach_gates.py`:
  - new script to sweep `attach_radius x attach_capture_min_points`,
  - reports ranked metrics and optional JSON/CSV outputs.
- `ip/generation/README.md`:
  - documented the new tuning command and `attach_capture_min_points`.

### Why
- We needed a clean, repeatable way to tune thin-object attach behavior without repeatedly changing logic.
- Metrics from event-level attach stats allow principled threshold selection while preserving paper-style kinematic generation.

### Follow-up
- The initial sweep measured mostly easy-positive attach recall.
- We extended tuning to include hard-negative probes at targeted close events (local-frame offsets around the close pose) and added `hard_negative_false_rate` to ranking.

## Runner Collision Audit (2026-02-11)

### Decision
- Harden apptainer runner/process orchestration against cross-run interference and false kills.

### Changed
- `apptainer/run_instant_policy_vnc.sh`:
  - replaced stale-process lookup with `pgrep -f` and removed global `fluxbox` kill pattern,
  - changed stale cleanup to TERM then KILL (instead of immediate `kill -9`).
- `apptainer/train_instant_policy.slurm`:
  - fixed `GEN_NUM_SHARDS=0` behavior (now truly disables generator workers),
  - made display/port allocation job-unique via `DISPLAY_BASE`/`VNC_BASE`,
  - shard and trainer runners now get non-overlapping `RLBENCH_DISPLAY`/`RLBENCH_VNC_PORT`.
- `apptainer/convert_peract.slurm`, `apptainer/generate_rlbench_data.slurm`, `apptainer/train_language.slurm`:
  - set job-/array-scoped display and VNC port defaults to avoid collisions in concurrent jobs.

### Why
- Failures with `line 91 ... ps|awk ... Killed` and early `apptainer exec ... Killed` were consistent with runner-level process collisions, not model/runtime logic.

### User-visible Effect
- Concurrent container launches no longer fight over display `:1` by default.
- `GEN_NUM_SHARDS=0` works as an actual disable switch.

## MSI Training Pipeline Rewrite (2026-02-11)

### Decision
- Replace full-ring bootstrap with minimal bootstrap + continuous sharded generation during training.
- Keep one SLURM entrypoint script, but make generation truly parallel by running shard workers in background.

### Changed
- `apptainer/train_instant_policy.slurm`:
  - defaults: `USE_WANDB=1`, `AUTO_RESUME=1`,
  - minimal bootstrap target: `MIN_BOOTSTRAP_TASKS=512`,
  - continuous producer enabled by default:
    - `ENABLE_PARALLEL_GENERATORS=1`
    - `GEN_NUM_SHARDS=4`
    - `GEN_CHUNK_TASKS=256`
  - generators shard over active ring files visible to the current dataloader run (`ACTIVE_BUFFER_SIZE=train_count at train start`),
  - per-shard logs emitted to `/scratch.global/$USER/ips/logs/ip_gen_<jobid>_shard*.log`,
  - cleanup trap stops generator workers when the training job exits.

### Why
- First-principles producer/consumer balance: avoid waiting many hours for full upfront generation.
- Keep training fed with fresh pseudo-demos while preserving fixed validation and checkpoint resume.

### User-visible Effect
- Startup latency is much lower than full prefill.
- Data refresh runs during training by default.

## Training Throughput Controls (2026-02-11)

### Decision
- Do not lock runtime behavior to paper defaults; expose throughput knobs for MSI tuning.

### Changed
- `ip/train.py`:
  - made `--compile_models` effective for scratch/fine-tune/resume paths.
  - added CLI controls:
    - `--num_workers`
    - `--persistent_workers`
    - `--prefetch_factor`
    - `--val_check_interval`
    - `--log_every_n_steps`
    - `--devices`
    - `--strategy`
- `apptainer/train_instant_policy.slurm`:
  - defaults tuned for speed:
    - `COMPILE_MODELS=1`
    - `TRAIN_NUM_WORKERS=16`
    - `TRAIN_PREFETCH_FACTOR=4`
    - `TRAIN_VAL_CHECK_INTERVAL=50000`
  - passes new throughput args into `ip/train.py`.

### Why
- Fastest training depends on hardware and pipeline saturation, not strict paper reproduction.

### User-visible Effect
- Throughput can be tuned from SLURM env vars without editing Python.

## Hardware Telemetry for 1h Profiling (2026-02-11)

### Decision
- Add explicit hardware/process telemetry logging inside the unified training SLURM pipeline.

### Changed
- `apptainer/train_instant_policy.slurm` now starts a background sampler that writes:
  - host CPU utilization and RAM usage,
  - GPU utilization/VRAM/power/temp (from `nvidia-smi`),
  - trainer process RSS/%CPU,
  - generator worker count + aggregate RSS/%CPU,
  - periodic train ring task count.
- generator shard logs now include unix timestamps per chunk line (`ts=<epoch>`), enabling exact first-hour throughput analysis.
- Telemetry output path:
  - `/scratch.global/$USER/ips/logs/ip_hw_<jobid>.csv`
- Added env knobs:
  - `TELEMETRY_INTERVAL_SEC` (default `30`)
  - `TELEMETRY_TASK_COUNT_EVERY` (default `10` samples)
- Cleanup trap now stops both generator workers and telemetry sampler.

### Why
- W&B/system metrics can be unavailable or incomplete during cluster runs.
- We need deterministic 1-hour evidence to pick shard count and dataloader settings.

### User-visible Effect
- Every run now produces a machine-readable telemetry CSV usable for shard/throughput decisions.

## DataLoader Segfault Stabilization (2026-02-11)

### Decision
- Remove `open3d` from DataLoader worker import path and use a pure NumPy point-cloud subsampling path.

### Changed
- `ip/utils/common_utils.py`
  - removed top-level `open3d` import.
  - moved `open3d` import inside `downsample_pcd` only.
- `ip/utils/data_proc.py`
  - removed top-level `open3d` import.
  - `subsample_pcd` now does: finite-point filtering + random sampling (with replacement when needed).
  - `remove_statistical_outliers` now returns finite-point filtering (no Open3D call).
- `ip/train.py`
  - sets `torch.set_float32_matmul_precision('high')` at startup for Tensor Core throughput and to remove Lightning warning.

### Why
- Training DataLoader workers (`TrajectoryDataset`) import `data_proc/common_utils`.
- Top-level Open3D import in forked workers is a common source of worker segfault instability.
- Pseudo-demo point clouds are already synthetic/clean; expensive Open3D outlier removal in training hot path is unnecessary.

### User-visible Effect
- Eliminates Open3D-related worker segfault risk in training dataloaders.
- Lower per-sample CPU overhead during training.

## Resume/Env Robustness Fixes (2026-02-11)

### Decision
- Prevent first-run crashes when `AUTO_RESUME=1` and no checkpoint exists.
- Remove deprecated Transformers cache env usage.

### Changed
- `apptainer/train_instant_policy.slurm`:
  - `--auto_resume` is now passed only if `<save_root>/<run_name>` already contains a `.pt` checkpoint.
  - otherwise logs a clear message and starts fresh.
- `apptainer/run_instant_policy_vnc.sh`:
  - removed `TRANSFORMERS_CACHE` export and uses `HF_HOME` only.

### Why
- New run names should not fail resume logic.
- Avoid noisy deprecation warning from `transformers`.

### User-visible Effect
- First submission with a new `RUN_NAME` starts cleanly.
- Re-submissions with existing checkpoints still auto-resume.
