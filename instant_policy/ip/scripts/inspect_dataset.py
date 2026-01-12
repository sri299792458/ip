import argparse
import glob
import os
import random

import numpy as np
import torch

from ip.utils.common_utils import transforms_to_actions


def summarize(values, name):
    values = np.asarray(values)
    if values.size == 0:
        print(f'{name}: no samples')
        return
    print(f'{name}: mean={values.mean():.6f} med={np.median(values):.6f} '
          f'p95={np.percentile(values, 95):.6f} max={values.max():.6f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with data_*.pt files')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of files to sample for stats')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None, help='Optional output dir for sample .npz files')
    parser.add_argument('--save_count', type=int, default=5, help='How many samples to save')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, 'data_*.pt')))
    if not files:
        raise RuntimeError(f'No data_*.pt files found in {args.data_dir}')

    random.seed(args.seed)
    sample_count = min(args.num_samples, len(files))
    sample_files = random.sample(files, sample_count)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    trans_norms = []
    rot_norms = []
    grip_vals = []
    obs_sizes = []

    saved = 0
    for path in sample_files:
        data = torch.load(path)

        actions = data.actions
        if actions.dim() == 4:
            actions = actions[0]
        actions_6d = transforms_to_actions(actions)
        trans = actions_6d[..., :3]
        rot = actions_6d[..., 3:]

        trans_norms.extend(torch.norm(trans, dim=-1).cpu().numpy().tolist())
        rot_norms.extend(torch.norm(rot, dim=-1).cpu().numpy().tolist())

        if hasattr(data, 'actions_grip'):
            grip = data.actions_grip
            if grip.dim() > 1:
                grip = grip.view(-1)
            grip_vals.extend(grip.cpu().numpy().tolist())

        if hasattr(data, 'pos_obs'):
            obs_sizes.append(int(data.pos_obs.shape[0]))

        if args.save_dir and saved < args.save_count:
            out_path = os.path.join(args.save_dir, f'sample_{saved}.npz')
            payload = {
                'pos_obs': data.pos_obs.cpu().numpy() if hasattr(data, 'pos_obs') else None,
                'actions_6d': actions_6d.cpu().numpy(),
                'current_grip': data.current_grip.cpu().numpy() if hasattr(data, 'current_grip') else None,
                'actions_grip': data.actions_grip.cpu().numpy() if hasattr(data, 'actions_grip') else None,
            }
            if hasattr(data, 'pos_demos') and data.pos_demos is not None:
                payload['pos_demos'] = data.pos_demos.cpu().numpy()
            np.savez(out_path, **payload)
            saved += 1

    print(f'Scanned {sample_count} / {len(files)} samples from {args.data_dir}')
    summarize(trans_norms, 'translation_norm (meters)')
    summarize(np.degrees(rot_norms), 'rotation_norm (degrees)')
    summarize(grip_vals, 'actions_grip (raw)')
    if obs_sizes:
        summarize(obs_sizes, 'num_points pos_obs')

    if args.save_dir:
        print(f'Saved {saved} samples to {args.save_dir}')


if __name__ == '__main__':
    main()
