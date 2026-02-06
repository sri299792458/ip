# Deployment Running Notes

Last updated: 2026-02-06

## Log Discipline

### Decision
Keep this file as the canonical deployment decision log.

### Rule
- Any behavior change in `ip/deployment` gets a short note here in the same work session.
- Each note must include: what changed, why, and the user-visible effect.

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
