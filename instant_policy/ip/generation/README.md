# Pseudo Demonstration Generation

This module generates pseudo demonstrations described in the Instant Policy paper.

## Quick start

```bash
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path /path/to/shapenet \
  --save_dir ./data/pseudo_demos \
  --num_tasks 1000
```

For cluster runs, use `--task_start` and `--num_tasks` per shard, then merge:

```bash
python -m ip.scripts.merge_pseudo_demos \
  --input_dirs /scratch/pseudo_demos/shard_* \
  --output_dir ./data/pseudo_demos_merged
```
