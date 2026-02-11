# Bimanual Pseudo-Demo Generation

This module generates bimanual pseudo-training samples for `ip.train_bimanual` with the same asset philosophy as single-arm Instant Policy pseudo data:
- ShapeNet object scenes,
- canonicalized Robotiq 2F-85 gripper mesh,
- object-centric attach on close transitions,
- kinematic trajectories (not dynamic simulation).

## Task Set

Default sampling uses the 13 RLBench2 bimanual tasks:
- `bimanual_push_box`
- `bimanual_lift_ball`
- `bimanual_dual_push_buttons`
- `bimanual_pick_plate`
- `bimanual_put_item_in_drawer`
- `bimanual_put_bottle_in_fridge`
- `bimanual_handover_item`
- `bimanual_pick_laptop`
- `bimanual_straighten_rope`
- `bimanual_sweep_to_dustpan`
- `bimanual_lift_tray`
- `bimanual_handover_item_easy`
- `bimanual_take_tray_out_of_oven`

Variation counts are encoded in `config.py` (`PERACT2_BIMANUAL_VARIATIONS`, sum = 23).

## Primitive Design

The 13 tasks are generated from 7 reusable primitive families:
- `cooperative_lift`
- `dual_push_sync`
- `dual_push_transport`
- `container_open_place_remove`
- `handover`
- `two_endpoint_tension`
- `tool_plus_receptacle`

## Gripper/Object Semantics

- Gripper state convention: `1=open`, `0=closed`.
- Attach decision: evaluated only on open->closed transition.
- Targeted attach: only the targeted object index is eligible for that arm.
- Capture test: object surface points inside the estimated jaw-capture region of the canonical gripper mesh.
- Detach: immediate on closed->open transition.
- If both arms contend for the same object, the later successful close transition owns attachment.

## Output Contract

Each `task_*.pt` file is `BimanualWorldBatch` compatible and contains:
- `points_world` `[1, N, 3]`
- `T_w_left_current` `[1, 4, 4]`
- `T_w_right_current` `[1, 4, 4]`
- `T_w_left_future` `[1, P, 4, 4]`
- `T_w_right_future` `[1, P, 4, 4]`
- `grip_left_current` `[1]`
- `grip_right_current` `[1]`
- `grip_left_future` `[1, P]`
- `grip_right_future` `[1, P]`

## Commands

Generate training samples:

```bash
python -m ip.scripts.generate_bimanual_pseudo_demos \
  --shapenet_path /workspace/data/shapenet \
  --gripper_mesh_path /workspace/data/assets/robotiq_2f85_collision_open.obj \
  --shapenet_index_path /workspace/data/pseudo_ring/shapenet_index.json \
  --save_dir /workspace/data/pseudo_bimanual/train \
  --num_samples 10000 \
  --pred_horizon 8 \
  --num_points 2048
```

Debug one task with video only:

```bash
python -m ip.scripts.generate_bimanual_pseudo_demos \
  --shapenet_path /workspace/data/shapenet \
  --gripper_mesh_path /workspace/data/assets/robotiq_2f85_collision_open.obj \
  --save_dir /workspace/data/pseudo_bimanual/debug_handover \
  --num_samples 8 \
  --forced_task bimanual_handover_item \
  --render_make_videos \
  --render_video_dir /workspace/data/pseudo_bimanual/debug_handover/videos \
  --render_visual_width 800 \
  --render_visual_height 800
```

Ring-buffer mode:

```bash
python -m ip.scripts.generate_bimanual_pseudo_demos \
  --shapenet_path /workspace/data/shapenet \
  --gripper_mesh_path /workspace/data/assets/robotiq_2f85_collision_open.obj \
  --save_dir /workspace/data/pseudo_bimanual/ring \
  --num_samples 999999 \
  --buffer_size 8192
```
