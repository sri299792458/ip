# Running Notes: Instant Policy + Video-Derived Context

Date: 2026-03-03  
Workspace: `/home/srinivas/Desktop/ip`

## 1) Goal

Replace the real-world one-demo context in Instant Policy with context reconstructed from video:
- point clouds from Spatial Tracker v2 + Depth Anything 2
- hand pose from HAMER
- gripper state from `out_gripper_status.npy`

Key challenge: hand-to-Robotiq-2F85 kinematic retargeting (pose + gripper semantics).

## 2) Instant Policy Input Contract (paper + code)

From paper + `ip/utils/data_proc.py`, deployment docs:
- raw demo format:
  - `pcds`: list of segmented object point clouds in world/base frame (`N x 3`)
  - `T_w_es`: list of end-effector SE(3) poses (`4 x 4`) in same frame
  - `grips`: binary gripper state per frame
- context demo fed to model is waypointed (`L=10`) by `extract_waypoints(...)`, which prioritizes:
  - start/end
  - gripper transitions + pre-transition frames
  - SE(3) arc-length coverage
- this repo currently uses RLBench gripper convention:
  - `1 = open`, `0 = closed`

Important implication:
- For adapter compatibility, we need **full SE(3)** per frame, not only wrist translation.

## 3) Current Data Inspection (`rpm_apple`)

Files:
- `out_poses_in_base.npy`
- `out_gripper_status.npy`

I added a reusable inspector:
- `rpm_apple/analyze_context_npy.py`

Run with:
```bash
conda run --no-capture-output -n ip_env python rpm_apple/analyze_context_npy.py --one-means-closed
```

Observed:
- both files contain `150` records (frame indices `0..149`)
- `out_poses_in_base.npy` payload:
  - dict keys: `frame_idx`, `hand_pose`, `wrist_pose`, `object_pose`
  - `wrist_pose`: mostly `(1,3)`, `None` in 51 frames, one NaN frame at 110
  - `hand_pose`: `(21,3)` when present, but with many NaNs per frame (partial keypoint dropouts)
  - `object_pose`: **`(N,6)` not `(N,3)`**; columns appear to be `xyzrgb`
- present data interval:
  - most usable frames: `11..107`, plus `109`; frame `110` has NaN wrist

Gripper timeline (`out_gripper_status.npy`, raw):
- values: `{0,1}`
- transitions at frames `30` and `91`
- segments: `[0..29]=0`, `[30..90]=1`, `[91..149]=0`

You stated raw semantics are:
- `1 = closed`, `0 = open`

That means raw timeline is:
- open -> closed -> open

## 4) Key Mismatches To Fix Before Instant Policy Adapter

1. **Object cloud dimensionality mismatch**
- Instant Policy expects geometry-only points (`N x 3`).
- Current `object_pose` is `N x 6`.
- Adapter must slice `object_pose[:, :3]`.

2. **Gripper semantics mismatch**
- Your raw status: `1=closed, 0=open`
- Instant Policy training/deployment path here expects RLBench style (`1=open, 0=closed`)
- Adapter should invert:
  - `grip_rlbench = 1 - grasp_status_raw`

3. **Pose completeness mismatch**
- We only have wrist translation directly.
- Need full orientation to build `T_w_e`.
- Must recover orientation from hand keypoints (with robust fallback + smoothing).

4. **NaN/dropout handling**
- Hand keypoints are sparse/noisy in several joints.
- Adapter needs temporal imputation and confidence/fallback logic.

## 5) Hand Keypoint Quality Snapshot

From available `(21,3)` hand frames:
- stable joints (high availability): `0,1,2,3,5` (~99%)
- many distal joints have low availability
- `wrist_pose` is numerically equal to `hand_pose[0]` when finite

This is good news: wrist anchor is consistent.

## 6) Retargeting Strategy (pragmatic first pass)

### Pass A: Build a robust SE(3) estimate from sparse hand keypoints

Candidate construction (per frame):
- origin:
  - preferred: midpoint of thumb/index reference points
  - fallback: wrist (`joint 0`) if tips missing
- orientation:
  - define palm basis from robust MCP-level points (e.g., joints `0,2,5,17` with fallbacks)
  - construct orthonormal frame via cross products
  - enforce temporal sign continuity for axes

Then map hand frame to Robotiq policy frame using constant extrinsic:
- `R_gripper = R_hand @ R_offset`
- `t_gripper = t_hand + R_hand @ t_offset`
- `(R_offset, t_offset)` calibrated from a short alignment segment (or one manually selected frame).

### Pass B: Use provided gripper status directly

- avoid inferring open/close from finger distances initially
- convert raw to RLBench convention via inversion
- use transitions to preserve waypoint anchors in `extract_waypoints`

### Pass C: Build demo object for Instant Policy

Output raw sample:
- `pcds[t] = object_pose[t][:, :3]` (after NaN filtering)
- `T_w_es[t] = estimated 4x4 from Pass A`
- `grips[t] = 1 - grasp_status_raw[t]` (if raw is 1=closed)

Then call existing converter path:
- `sample_to_cond_demo(sample, num_waypoints=10)`

## 7) Immediate Risks

1. Orientation ambiguity from sparse/noisy joints can destabilize SE(3).
2. Frame convention mismatch (hand/object may already be in base frame; do not re-apply transforms blindly).
3. If `HAT_TO_BASE` is already applied (filename says “in_base”), double-transforming will break geometry.
4. Gripper-state convention inversion must be done exactly once.

## 8) Next Concrete Work Items

1. Implement adapter script:
- read both `.npy`
- sanitize frames
- estimate `T_w_e`
- emit `demo_from_video.pkl` (`pcds`, `T_w_es`, `grips`, plus debug metadata)

2. Add debug outputs:
- per-frame axis visualization values
- dropped-frame log with reasons
- transition frame report (`30`, `91` expected in this sample)

3. Validate with Instant Policy utilities:
- `sample_to_cond_demo(...)`
- `ip.deployment.utils.inspect_demo --demo <generated.pkl>`
- optionally `view_demo_pcds`

## 9) Open Questions For Alignment

1. Confirm final gripper convention at your source is exactly `1=closed, 0=open` for all files.
2. Confirm whether `out_poses_in_base.npy` is already in robot base frame (I assume yes from name).
3. Decide calibration style for hand->gripper extrinsic:
- one manual frame
- short calibration sequence
- optimization against object-contact events.

## 10) New Clarification (2026-03-03)

You confirmed:
- the current `.npy` sample came from a **real demo**
- target direction is to extract equivalent context from **video generated by models (e.g., Veo, Wan)**

Implication:
- the real demo should be treated as a **reference distribution** for sanity checks
- generated-video extraction should be treated as a **domain-shifted data source** that must be normalized to the same Instant Policy contract.

## 11) Veo/Wan-Oriented Adapter Requirements

When context comes from generated video, the top additional failure modes are:
1. Metric scale ambiguity/drift in depth.
2. Camera intrinsics/extrinsics mismatch across frames.
3. Temporal inconsistency (object geometry flicker, hand jitter, topology glitches).
4. Hand anatomy artifacts that break keypoint-derived orientation.
5. Contact-stage inconsistency relative to provided gripper labels.

Concrete countermeasures to add in adapter:
1. **Metric stabilization**
- normalize depth scale by anchoring to object-size prior or hand bone-length prior before generating `T_w_es`.

2. **Temporal filtering**
- smooth hand keypoints and wrist trajectory with robust filtering (median + Savitzky-Golay / EMA).
- smooth orientation on SO(3) with quaternion continuity enforcement.

3. **Geometry stabilization**
- downsample and denoise object clouds per frame, then enforce temporal overlap consistency checks.

4. **Quality gating**
- per-frame confidence score from finite-joint count + orientation condition number + object point count.
- drop or impute low-confidence frames before waypoint extraction.

5. **Contract-finalization**
- emit exactly `pcds (N,3)`, `T_w_es (4,4)`, `grips (RLBench convention)` and validate with `sample_to_cond_demo`.

## 12) Waypoint Selection vs NaNs (code-level conclusion)

Reviewed implementation:
- `extract_waypoints(...)` in `instant_policy/ip/utils/data_proc.py`
- key behavior:
  - start/end are always anchors
  - gripper transition frames and pre-transition frames are prioritized
  - remaining points are selected by SE(3) motion coverage

Practical conclusion:
1. It is true that only `L=10` waypoint frames are ultimately used for context.
2. But NaNs are only harmless if they are in frames that are neither:
- selected as waypoints, nor
- needed as trajectory endpoints/transition anchors.
3. NaNs in first/last frame are risky because endpoints are forced anchors.
4. NaNs near gripper transitions are risky because transition anchors are prioritized.

Observed on current finite subset (identity-rotation proxy from wrist translation):
- selected original frames: `[11, 29, 30, 46, 55, 69, 88, 91, 101, 109]`
- this confirms transition-adjacent frames (`30`, `91`) are indeed selected.

Action for adapter:
- pre-filter to finite-valid frames before running waypoint extraction
- ensure first/last of the filtered sequence are finite and meaningful.

## 13) Hand Landmark Reference (21 joints)

Local reference image:
- `/home/srinivas/Desktop/ip/assets/reference_images/mediapipe_hand_21_landmarks.png`
- `assets/reference_images/mediapipe_hand_21_landmarks.png`

Embedded preview:

![MediaPipe Hand Landmarks (21 joints)](assets/reference_images/mediapipe_hand_21_landmarks.png)

Official source:
- `https://mediapipe.dev/images/mobile/hand_landmarks.png`

## 14) Generated MVP demo.pkl (2026-03-03)

Generated file:
- `/home/srinivas/Desktop/ip/rpm_apple/demo.pkl`

Generator script:
- `/home/srinivas/Desktop/ip/rpm_apple/generate_demo_pkl_from_npy.py`

Command used:
```bash
conda run --no-capture-output -n ip_env \
  python /home/srinivas/Desktop/ip/rpm_apple/generate_demo_pkl_from_npy.py
```

Current conversion assumptions:
1. Apply `HAT_TO_BASE` from provided snippet to wrist and object points.
2. Use fixed top-down orientation for all `T_w_es`:
- `R = diag(1, -1, -1)` (Rx(pi)).
3. Use wrist translation as proxy end-effector translation.
4. Convert grasp labels from raw (`1=closed`) to RLBench convention (`1=open`) via inversion.
5. Keep only finite-valid frames (drop frames with missing/non-finite wrist or invalid object cloud).

Output summary:
- frames kept: `98` (from original indices `11..109`, excluding invalids)
- dropped:
  - `missing_wrist=51`
  - `nonfinite_wrist=1`
- output keys:
  - `pcds`, `T_w_es`, `grips`, `frame_spec`, `recorded_at_utc`, `source_meta`

Validation run:
```bash
conda run --no-capture-output -n ip_env \
  python -m ip.deployment.utils.inspect_demo --demo /home/srinivas/Desktop/ip/rpm_apple/demo.pkl
```

Result:
- waypoint extraction succeeded
- 10 selected waypoint frames include transition-adjacent anchors.

## 15) Trajectory Playback (viser)

Use the repo's built-in interactive viewer:
- `instant_policy/ip/deployment/utils/view_demo_pcds.py`

Command:
```bash
conda run --no-capture-output -n ip_env \
  python -m ip.deployment.utils.view_demo_pcds \
  --demo /home/srinivas/Desktop/ip/rpm_apple/demo.pkl
```

Notes:
- This visualizes object point clouds in the end-effector frame over time.
- The previously generated offline GIF preview was removed by request.

## 16) Source Video Provenance

You confirmed this dataset was extracted from generated video.

Reference file:
- `/home/srinivas/Desktop/ip/rpm_apple/other/camera_visualization.mp4`

Quick metadata:
- resolution: `640 x 476`
- fps: `15`
- frames: `99`
- duration: `~6.6s`

## 17) Transform Verification Plan (to resolve frame confusion)

Key principle:
- If wrist and object are both transformed by the same rigid transform, many internal relative metrics
  (wrist-object distance, motion correlation) stay invariant.
- So transform ambiguity cannot be resolved reliably from those relative metrics alone.

Better verification method:
1. Collect at least 3 corresponding 3D anchor points:
- points in `hat` frame (from tracker/pcd pipeline),
- matching points in robot `base` frame (measured or touched in robot space).
2. Fit rigid transform `HAT_TO_BASE` from those correspondences.
3. Re-generate `demo.pkl` using the fitted transform and replay-check.

Added utility:
- `/home/srinivas/Desktop/ip/rpm_apple/fit_hat_to_base_from_pairs.py`

Usage:
```bash
python /home/srinivas/Desktop/ip/rpm_apple/fit_hat_to_base_from_pairs.py \
  --pairs-json /path/to/hat_base_pairs.json \
  --out-matrix-txt /home/srinivas/Desktop/ip/rpm_apple/hat_to_base_fitted.txt \
  --print-example-cmd
```

Generator now supports custom transform matrix:
- `/home/srinivas/Desktop/ip/rpm_apple/generate_demo_pkl_from_npy.py`
- new arg:
  - `--hat-to-base-matrix-txt <4x4_txt>`
  - overrides `--hat-to-base-mode {none,forward,inverse}`.
