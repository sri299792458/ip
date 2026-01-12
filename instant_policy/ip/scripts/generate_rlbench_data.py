import argparse
import os
import sys
import numpy as np
import torch

# Ensure `ip` package is importable when running as a script.
script_dir = os.path.dirname(os.path.abspath(__file__))
instant_policy_root = os.path.dirname(os.path.dirname(script_dir))
if instant_policy_root not in sys.path:
    sys.path.insert(0, instant_policy_root)

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from ip.models.scene_encoder import SceneEncoder
from ip.utils.rl_bench_tasks import TASK_NAMES
from ip.utils.rl_bench_utils import rl_bench_demo_to_sample
from ip.utils.data_proc import sample_to_cond_demo, sample_to_live, save_sample


def setup_env(task_name, headless=True, restrict_rot=True):
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    env = Environment(
        action_mode,
        './',
        obs_config=obs_config,
        headless=headless
    )
    env.launch()

    if task_name not in TASK_NAMES:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_NAMES.keys())}")

    task_class = TASK_NAMES[task_name]
    task = env.get_task(task_class)

    def temp(position, euler=None, quaternion=None, ignore_collisions=False, trials=300, max_configs=1,
             distance_threshold=0.65, max_time_ms=10, trials_per_goal=1, algorithm=None, relative_to=None):
        return env._robot.arm.get_linear_path(position, euler, quaternion, ignore_collisions=ignore_collisions,
                                              relative_to=relative_to)

    env._robot.arm.get_path = temp
    env._scene._start_arm_joint_pos = np.array([
        6.74760377e-05, -1.91104114e-02, -3.62065766e-05, -1.64271665e+00,
        -1.14094291e-07, 1.55336857e+00, 7.85427451e-01
    ])

    if restrict_rot:
        rot_bounds = env._scene.task.base_rotation_bounds()
        mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
        env._scene.task.base_rotation_bounds = lambda: (
            (0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
            (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3))
        )

    return env, task


def collect_demo_sample(task, max_attempts):
    for _ in range(max_attempts):
        try:
            demos = task.get_demos(1, live_demos=True, max_attempts=1000)
            return rl_bench_demo_to_sample(demos[0])
        except Exception:
            continue
    raise RuntimeError('Failed to collect a demo after max attempts.')


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
    parser.add_argument('--task_name', type=str, required=True, help='RLBench task name')
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of episodes to generate')
    parser.add_argument('--num_demos', type=int, default=2, help='Number of context demos per episode')
    parser.add_argument('--num_waypoints_demo', type=int, default=10, help='Waypoints per demo')
    parser.add_argument('--pred_horizon', type=int, default=8, help='Prediction horizon')
    parser.add_argument('--num_points', type=int, default=2048, help='Points per point cloud')
    parser.add_argument('--save_dir', type=str, default='./data/train', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device for scene encoder')
    parser.add_argument('--headless', action='store_true', help='Run RLBench headless')
    parser.add_argument('--restrict_rot', action='store_true', help='Restrict base rotation bounds')
    parser.add_argument('--max_demo_attempts', type=int, default=50, help='Attempts per demo')
    parser.add_argument('--compute_embeddings', action='store_true', help='Precompute scene embeddings')
    parser.add_argument('--scene_encoder_path', type=str, default='./checkpoints/scene_encoder.pt',
                        help='Path to scene encoder checkpoint')
    parser.add_argument('--live_spacing_trans', type=float, default=0.01, help='Translation spacing for live data')
    parser.add_argument('--live_spacing_rot', type=float, default=3.0, help='Rotation spacing (degrees) for live data')
    parser.add_argument('--subsample_live', action='store_true', help='Subsample live trajectories')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    scene_encoder = None
    if args.compute_embeddings:
        scene_encoder = SceneEncoder(num_freqs=10, embd_dim=512)
        scene_encoder.load_state_dict(torch.load(args.scene_encoder_path))
        scene_encoder = scene_encoder.to(args.device)
        scene_encoder.eval()

    env, task = setup_env(args.task_name, headless=args.headless, restrict_rot=args.restrict_rot)

    offset = next_offset(args.save_dir)
    total_saved = 0

    for ep in range(args.num_episodes):
        demo_samples = []
        for _ in range(args.num_demos + 1):
            demo_samples.append(collect_demo_sample(task, args.max_demo_attempts))

        cond_demos = [
            sample_to_cond_demo(sample, args.num_waypoints_demo, num_points=args.num_points)
            for sample in demo_samples[:args.num_demos]
        ]
        live = sample_to_live(demo_samples[-1],
                              args.pred_horizon,
                              num_points=args.num_points,
                              trans_space=args.live_spacing_trans,
                              rot_space=args.live_spacing_rot,
                              subsample=args.subsample_live)

        full_sample = {
            'demos': cond_demos,
            'live': live,
        }

        save_sample(full_sample, save_dir=args.save_dir, offset=offset, scene_encoder=scene_encoder)
        num_saved = len(live['obs'])
        offset += num_saved
        total_saved += num_saved
        print(f'Episode {ep + 1}/{args.num_episodes}: saved {num_saved} samples.')

    env.shutdown()
    print(f'Finished. Total saved samples: {total_saved}')


if __name__ == '__main__':
    main()
