# Bimanual Pseudo-Demo Generation

This module generates bimanual pseudo-training samples for `ip.train_bimanual`.

## Goal

Mirror the PerAct2 RLBench bimanual task family structure, while staying in Instant Policy's pseudo-data philosophy:
- kinematic trajectory synthesis,
- object-centric scene/context,
- broad pretraining distribution before task fine-tuning.

## Task Set

Default task sampling uses the 13 RLBench2 bimanual tasks:
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

Benchmark variation counts are encoded in `config.py` (sum = 23).

## Primitive Design

The 13 tasks are generated from 7 reusable primitives:
- `cooperative_lift`
- `dual_push_sync`
- `dual_push_transport`
- `container_open_place_remove`
- `handover`
- `two_endpoint_tension`
- `tool_plus_receptacle`

This keeps generation compositional and avoids per-task hacks.

## Output Contract

Each file is a `BimanualWorldBatch`-compatible sample (`task_*.pt`) with keys:
- `points_world` `[1, N, 3]`
- `T_w_left_current` `[1, 4, 4]`
- `T_w_right_current` `[1, 4, 4]`
- `T_w_left_future` `[1, P, 4, 4]`
- `T_w_right_future` `[1, P, 4, 4]`
- `grip_left_current` `[1]`
- `grip_right_current` `[1]`
- `grip_left_future` `[1, P]`
- `grip_right_future` `[1, P]`

Gripper convention follows Instant Policy/RLBench style:
- `1 = open`, `0 = closed`

## Commands

Generate 1000 samples on all tasks:

```bash
python -m ip.scripts.generate_bimanual_pseudo_demos \
  --save_dir /path/to/pseudo_bimanual/train \
  --num_samples 1000 \
  --pred_horizon 8 \
  --num_points 2048
```

Force one task type (debug):

```bash
python -m ip.scripts.generate_bimanual_pseudo_demos \
  --save_dir /path/to/pseudo_bimanual/debug_handover \
  --num_samples 64 \
  --forced_task bimanual_handover_item
```

Ring-buffer mode:

```bash
python -m ip.scripts.generate_bimanual_pseudo_demos \
  --save_dir /path/to/pseudo_bimanual/ring \
  --num_samples 999999 \
  --buffer_size 8192
```
