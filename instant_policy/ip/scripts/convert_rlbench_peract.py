import argparse
import io
import os
import random
import zipfile

import numpy as np
import torch

from ip.models.scene_encoder import SceneEncoder
from ip.utils.common_utils import pose_to_transform
from ip.utils.data_proc import sample_to_cond_demo, sample_to_live, save_sample


DEPTH_SCALE = 2 ** 24 - 1


def image_to_float_array(image_array, scale_factor):
    image_array = np.asarray(image_array)
    image_shape = image_array.shape
    channels = image_shape[2] if len(image_shape) > 2 else 1
    if channels == 3:
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    else:
        float_array = image_array.astype(np.float32)
    return float_array / scale_factor


def rgb_handles_to_mask(rgb_coded_handles):
    rgb = rgb_coded_handles * 255.0
    rgb = rgb.astype(np.int64)
    return (rgb[:, :, 0] +
            rgb[:, :, 1] * 256 +
            rgb[:, :, 2] * 256 * 256)


def _create_uniform_pixel_coords_image(resolution):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords


def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector, (h, w, -1))


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo


def pointcloud_from_depth_and_camera_params(depth, extrinsics, intrinsics):
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)
    c_vec = np.expand_dims(extrinsics[:3, 3], 0).T
    rot = extrinsics[:3, :3]
    rot_inv = rot.T
    rot_inv_c = np.matmul(rot_inv, c_vec)
    extrinsics = np.concatenate((rot_inv, -rot_inv_c), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(
        pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]
    return world_coords


def read_png(zip_file, name):
    try:
        import imageio.v2 as imageio
        return imageio.imread(io.BytesIO(zip_file.read(name)))
    except Exception:
        from PIL import Image
        return np.array(Image.open(io.BytesIO(zip_file.read(name))))


class Dummy:
    def __setstate__(self, state):
        self.__dict__.update(state)


class SafeUnpickler(torch.serialization.pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('rlbench') or module.startswith('pyrep'):
            return Dummy
        return super().find_class(module, name)


def load_observations(zip_file, pkl_path):
    data = SafeUnpickler(io.BytesIO(zip_file.read(pkl_path))).load()
    if isinstance(data, (list, tuple)):
        return data
    if hasattr(data, '_observations'):
        return data._observations
    raise RuntimeError(f'Could not unpack observations from {pkl_path}')


def list_episode_pkls(zip_file):
    return sorted([n for n in zip_file.namelist() if n.endswith('low_dim_obs.pkl')])


def resolve_frame_path(name_set, base, index):
    direct = f'{base}/{index}.png'
    if direct in name_set:
        return direct
    padded = f'{base}/{index:04d}.png'
    if padded in name_set:
        return padded
    prefixed = f'{base}/depth_{index:04d}.png'
    if prefixed in name_set:
        return prefixed
    prefixed = f'{base}/mask_{index:04d}.png'
    if prefixed in name_set:
        return prefixed
    raise FileNotFoundError(f'No frame image for {base} index {index}')


def episode_to_sample(zip_file, episode_root, cameras, mask_thresh, name_set):
    obs_list = load_observations(zip_file, f'{episode_root}/low_dim_obs.pkl')
    sample = {'pcds': [], 'T_w_es': [], 'grips': []}

    for idx, obs in enumerate(obs_list):
        pcds = []
        for cam in cameras:
            depth_base = f'{episode_root}/{cam}_depth'
            mask_base = f'{episode_root}/{cam}_mask'

            depth_path = resolve_frame_path(name_set, depth_base, idx)
            mask_path = resolve_frame_path(name_set, mask_base, idx)

            depth_img = read_png(zip_file, depth_path)
            mask_img = read_png(zip_file, mask_path)

            depth_norm = image_to_float_array(depth_img, DEPTH_SCALE)
            near = float(obs.misc[f'{cam}_camera_near'])
            far = float(obs.misc[f'{cam}_camera_far'])
            depth_m = near + depth_norm * (far - near)

            intr = obs.misc[f'{cam}_camera_intrinsics']
            extr = obs.misc[f'{cam}_camera_extrinsics']

            mask_rgb = mask_img.astype(np.float32) / 255.0
            mask_id = rgb_handles_to_mask(mask_rgb)
            mask = (mask_id > mask_thresh) & (depth_m > 0)
            if not np.any(mask):
                mask = depth_m > 0

            points = pointcloud_from_depth_and_camera_params(depth_m, extr, intr)
            points = points[mask]
            if points.size > 0:
                pcds.append(points)

        if not pcds:
            continue

        sample['pcds'].append(np.concatenate(pcds, axis=0))
        sample['T_w_es'].append(pose_to_transform(obs.gripper_pose))
        sample['grips'].append(float(obs.gripper_open))

    return sample


def next_offset(save_dir):
    if not os.path.exists(save_dir):
        return 0
    files = [f for f in os.listdir(save_dir) if f.startswith('data_') and f.endswith('.pt')]
    if not files:
        return 0
    indices = []
    for fname in files:
        try:
            indices.append(int(fname.split('_')[1].split('.')[0]))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_path', type=str, required=True, help='Path to RLBench task zip')
    parser.add_argument('--save_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=None, help='Limit number of episodes')
    parser.add_argument('--num_demos', type=int, default=2, help='Number of context demos')
    parser.add_argument('--num_waypoints_demo', type=int, default=10, help='Waypoints per demo')
    parser.add_argument('--pred_horizon', type=int, default=8, help='Prediction horizon')
    parser.add_argument('--num_points', type=int, default=2048, help='Points per point cloud')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for demo sampling')
    parser.add_argument('--mask_thresh', type=int, default=60, help='Mask threshold')
    parser.add_argument('--cameras', type=str,
                        default='front,left_shoulder,right_shoulder',
                        help='Comma-separated camera names')
    parser.add_argument('--compute_embeddings', action='store_true', help='Precompute scene embeddings')
    parser.add_argument('--scene_encoder_path', type=str, default='./checkpoints/scene_encoder.pt',
                        help='Path to scene encoder checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device for scene encoder')
    parser.add_argument('--live_spacing_trans', type=float, default=0.01, help='Translation spacing for live data')
    parser.add_argument('--live_spacing_rot', type=float, default=3.0, help='Rotation spacing (degrees) for live data')
    parser.add_argument('--subsample_live', action='store_true', help='Subsample live trajectories')
    parser.add_argument('--append', action='store_true', help='Append to existing output directory')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    existing = [f for f in os.listdir(args.save_dir) if f.startswith('data_') and f.endswith('.pt')]
    if existing and not args.append:
        raise RuntimeError(f'{args.save_dir} already has data_*.pt files. Use --append to continue.')

    offset = next_offset(args.save_dir) if args.append else 0
    rng = random.Random(args.seed)

    scene_encoder = None
    if args.compute_embeddings:
        scene_encoder = SceneEncoder(num_freqs=10, embd_dim=512)
        scene_encoder.load_state_dict(torch.load(args.scene_encoder_path))
        scene_encoder = scene_encoder.to(args.device)
        scene_encoder.eval()

    cameras = [c.strip() for c in args.cameras.split(',') if c.strip()]

    with zipfile.ZipFile(args.zip_path) as zf:
        name_set = set(zf.namelist())
        episode_pkls = list_episode_pkls(zf)
        if not episode_pkls:
            raise RuntimeError('No low_dim_obs.pkl files found in zip.')

        episode_roots = [p.rsplit('/', 1)[0] for p in episode_pkls]
        if args.num_episodes is not None:
            episode_roots = episode_roots[:args.num_episodes]

        demo_cache = {}
        total_saved = 0

        for ep_idx, live_root in enumerate(episode_roots):
            live_sample = episode_to_sample(zf, live_root, cameras, args.mask_thresh, name_set)

            if len(live_sample['pcds']) == 0:
                print(f'Skipping {live_root}: no valid frames.')
                continue

            live = sample_to_live(
                live_sample,
                args.pred_horizon,
                num_points=args.num_points,
                trans_space=args.live_spacing_trans,
                rot_space=args.live_spacing_rot,
                subsample=args.subsample_live
            )

            demo_roots = episode_roots[:]
            rng.shuffle(demo_roots)
            demo_roots = [r for r in demo_roots if r != live_root]
            if len(demo_roots) < args.num_demos:
                demo_roots = episode_roots[:]
                rng.shuffle(demo_roots)

            cond_demos = []
            for demo_root in demo_roots[:args.num_demos]:
                if demo_root in demo_cache:
                    cond_demo = demo_cache[demo_root]
                else:
                    demo_sample = episode_to_sample(zf, demo_root, cameras, args.mask_thresh, name_set)
                    cond_demo = sample_to_cond_demo(
                        demo_sample,
                        args.num_waypoints_demo,
                        num_points=args.num_points
                    )
                    demo_cache[demo_root] = cond_demo
                cond_demos.append(cond_demo)

            full_sample = {
                'demos': cond_demos,
                'live': live,
            }

            save_sample(full_sample, save_dir=args.save_dir, offset=offset, scene_encoder=scene_encoder)
            num_saved = len(live['obs'])
            offset += num_saved
            total_saved += num_saved
            print(f'Episode {ep_idx + 1}/{len(episode_roots)}: saved {num_saved} samples.')

        print(f'Finished. Total saved samples: {total_saved}')


if __name__ == '__main__':
    main()
