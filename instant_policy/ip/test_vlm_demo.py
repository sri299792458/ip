"""
Test script for VLM-generated synthetic demonstrations.

This script tests synthetic demonstrations from VLM-style outputs.
- Uses demo-derived outputs or task waypoints to synthesize 10-waypoint demos
- Actually runs the model with synthetic demos
- Compares task success rate: synthetic vs real demos
"""

import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import pickle

# Add instant_policy to path
script_dir = os.path.dirname(os.path.abspath(__file__))
instant_policy_root = os.path.dirname(script_dir)
if instant_policy_root not in sys.path:
    sys.path.insert(0, instant_policy_root)

from models.diffusion import GraphDiffusion
from utils.rl_bench_utils import rl_bench_demo_to_sample, get_point_cloud
from utils.vlm_to_demo import synthesize_demo_from_vlm
from utils.vlm_oracle import vlm_outputs_from_waypoints
from utils.data_proc import save_sample, sample_to_cond_demo, subsample_pcd
from utils.rl_bench_tasks import TASK_NAMES
from utils.common_utils import transform_to_pose, pose_to_transform, transform_pcd
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig


def extract_start_goal_from_demo(demo_sample):
    """Extract start and goal 3D positions from a real demonstration."""
    start_pose = demo_sample['T_w_es'][0]
    goal_pose = demo_sample['T_w_es'][-1]
    return start_pose[:3, 3], goal_pose[:3, 3]


def get_scene_pointcloud(rlbench_obs):
    """Extract combined point cloud from RLBench observation."""
    return get_point_cloud(rlbench_obs)


def setup_rlbench_env(task_name, headless=True, restrict_rot=True):
    """Setup RLBench environment for evaluation."""
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=headless
    )
    env.launch()
    if not headless:
        # Ensure GUI rendering is enabled under VNC/Xvfb
        if hasattr(env._pyrep, "set_rendering"):
            env._pyrep.set_rendering(True)
        if hasattr(env._pyrep, "step_ui"):
            env._pyrep.step_ui()

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


def run_model_rollout(
    model,
    task,
    demos,
    max_steps=30,
    execution_horizon=8,
    device='cuda'
):
    """
    Run model rollout with given demonstrations.
    Mirrors the logic from rl_bench_utils.py rollout_model.

    Returns:
        success: bool indicating task completion
    """
    # Reset environment
    descriptions, curr_obs = task.reset()

    # Cache demo scene embeddings (computed once per episode)
    demo_scene_node_embds = None
    demo_scene_node_pos = None

    # Initialize environment action
    env_action = np.zeros(8)

    # Convert gripper_pose (7-vector) to 4x4 transform matrix
    T_w_e = pose_to_transform(curr_obs.gripper_pose)

    for step_idx in range(max_steps):
        # Build current observation point cloud
        current_pcd_world = subsample_pcd(get_point_cloud(curr_obs))

        # Transform to end-effector frame (critical!)
        current_pcd_ee = transform_pcd(current_pcd_world, np.linalg.inv(T_w_e))

        # Construct sample dict for save_sample
        full_sample = {
            'demos': demos,
            'live': {
                'obs': [current_pcd_ee],  # EE-frame point cloud
                'grips': [curr_obs.gripper_open],
                'actions_grip': [np.zeros(8)],
                'T_w_es': [T_w_e],
                'actions': [T_w_e.reshape(1, 4, 4).repeat(8, axis=0)]
            }
        }

        # Build Data object
        data = save_sample(full_sample, None)

        # Compute scene embeddings
        if step_idx == 0:
            demo_scene_node_embds, demo_scene_node_pos = model.model.get_demo_scene_emb(
                data.to(device)
            )

        live_scene_node_embds, live_scene_node_pos = model.model.get_live_scene_emb(
            data.to(device)
        )

        # Attach embeddings to data
        data.live_scene_node_embds = live_scene_node_embds.clone()
        data.live_scene_node_pos = live_scene_node_pos.clone()
        data.demo_scene_node_embds = demo_scene_node_embds.clone()
        data.demo_scene_node_pos = demo_scene_node_pos.clone()

        # Run model inference
        with torch.no_grad():
            with torch.autocast(dtype=torch.float32, device_type=device):
                actions, grips = model.test_step(data.to(device), 0)
            actions = actions.squeeze().cpu().numpy()
            grips = grips.squeeze().cpu().numpy()

        # Execute actions in environment
        for j in range(execution_horizon):
            env_action[:7] = transform_to_pose(T_w_e @ actions[j])
            env_action[7] = int((grips[j] + 1) / 2 > 0.5)

            try:
                curr_obs, reward, terminate = task.step(env_action)
                success = int(terminate and reward > 0.)

                if terminate:
                    return bool(success)

            except Exception as e:
                print(f"      Action execution error: {e}")
                return False

        # Update end-effector pose for next iteration
        T_w_e = pose_to_transform(curr_obs.gripper_pose)

    # Max steps reached without success
    return False


def test_synthetic_demo_on_task(
    model,
    task_name,
    num_rollouts=10,
    use_synthetic=True,
    vlm_source='waypoints',
    device='cuda',
    headless=True,
    num_traj_wp=10,
    max_steps=30,
    execution_horizon=8
):
    """
    Test model with synthetic vs real demonstrations.

    Actually runs the model and measures task success.
    """
    env, task = setup_rlbench_env(task_name, headless=headless, restrict_rot=True)

    successes = 0

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    mode = 'Real Demos' if not use_synthetic else f"Synthetic Demos ({vlm_source})"
    print(f"Mode: {mode}")
    print(f"Rollouts: {num_rollouts}")
    print(f"{'='*60}\n")

    for rollout_idx in tqdm(range(num_rollouts), desc=f"Evaluating {task_name}"):
        try:
            if use_synthetic:
                print(f"\n  [Rollout {rollout_idx+1}] Generating synthetic demo...")

                if vlm_source == 'demo':
                    # Collect one real demo (oracle VLM outputs).
                    real_demo_raw = task.get_demos(1, live_demos=True, max_attempts=100)[0]
                    real_demo_sample = rl_bench_demo_to_sample(real_demo_raw)

                    # Extract start/goal from real demo
                    start_3d, goal_3d = extract_start_goal_from_demo(real_demo_sample)
                    scene_pcd = get_scene_pointcloud(real_demo_raw[0])

                    print(f"    Start: {start_3d}")
                    print(f"    Goal:  {goal_3d}")

                    # Build rich VLM outputs (mocked from real demo for now).
                    vlm_outputs = {
                        'poses_4x4': real_demo_sample['T_w_es'],
                        'trajectory_3d': [T[:3, 3] for T in real_demo_sample['T_w_es']],
                        'gripper_states': real_demo_sample['grips'],
                        'orientations': [T[:3, :3] for T in real_demo_sample['T_w_es']],
                        'pcd_sequence_world': real_demo_sample['pcds'],
                    }
                elif vlm_source == 'waypoints':
                    _, demo_obs = task.reset()
                    scene_pcd = get_scene_pointcloud(demo_obs)
                    vlm_outputs = vlm_outputs_from_waypoints(task, demo_obs)
                else:
                    raise ValueError(f"Unknown vlm_source: {vlm_source}")

                # Generate synthetic demo from rich outputs.
                synthetic_demo = synthesize_demo_from_vlm(
                    vlm_outputs,
                    current_pcd_world=scene_pcd,
                    num_waypoints=num_traj_wp,
                    interpolation_method='linear',
                    orientation_method='point_down'
                )

                print(f"    Generated {len(synthetic_demo['obs'])} waypoints")
                demos = [synthetic_demo]

            else:
                # Collect one real demo
                real_demo_raw = task.get_demos(1, live_demos=True, max_attempts=100)[0]
                real_demo_sample = rl_bench_demo_to_sample(real_demo_raw)

                # Use real demo - need to convert to conditioning format
                demos = [sample_to_cond_demo(real_demo_sample, num_waypoints=num_traj_wp)]

            # Actually run the model!
            print(f"    Running model rollout...")
            success = run_model_rollout(
                model=model,
                task=task,
                demos=demos,
                max_steps=max_steps,
                execution_horizon=execution_horizon,
                device=device
            )

            if success:
                successes += 1
                print(f"    Success!")
            else:
                print(f"    Failed")

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    success_rate = successes / num_rollouts
    print(f"\n{'='*60}")
    print(f"Final Success Rate: {success_rate:.1%} ({successes}/{num_rollouts})")
    print(f"{'='*60}\n")

    env.shutdown()
    return success_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='push_button',
                        help='RLBench task name')
    parser.add_argument('--model_path', type=str,
                        default='./checkpoints',
                        help='Path to model checkpoint directory')
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of evaluation rollouts')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--compare', action='store_true',
                        help='Compare synthetic vs real demos')
    parser.add_argument('--display', action='store_true',
                        help='Enable GUI rendering (not headless)')
    parser.add_argument('--vlm_source', type=str, default='waypoints',
                        choices=['demo', 'waypoints'],
                        help='Source of VLM-style outputs for synthetic demos')
    parser.add_argument('--num_traj_wp', type=int, default=10,
                        help='Number of waypoints per trajectory')
    parser.add_argument('--max_steps', type=int, default=30,
                        help='Max steps per rollout')
    parser.add_argument('--execution_horizon', type=int, default=8,
                        help='Action execution horizon')
    args = parser.parse_args()
    headless = not args.display

    # Load model using same approach as eval.py
    print(f"Loading model from {args.model_path}...")
    config = pickle.load(open(f'{args.model_path}/config.pkl', 'rb'))

    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_demos'] = 1
    config['num_diffusion_iters_test'] = 8

    model = GraphDiffusion.load_from_checkpoint(
        f'{args.model_path}/model.pt',
        config=config,
        strict=True,
        map_location=args.device
    ).to(args.device)

    model.model.reinit_graphs(1, num_demos=1)
    model.eval()
    print("Model loaded successfully!")

    if args.compare:
        print("\n" + "="*60)
        print("COMPARISON MODE: Synthetic vs Real Demos")
        print("="*60)

        synthetic_sr = test_synthetic_demo_on_task(
            model, args.task,
            num_rollouts=args.num_rollouts,
            use_synthetic=True,
            vlm_source=args.vlm_source,
            device=args.device,
            headless=headless,
            num_traj_wp=args.num_traj_wp,
            max_steps=args.max_steps,
            execution_horizon=args.execution_horizon
        )

        real_sr = test_synthetic_demo_on_task(
            model, args.task,
            num_rollouts=args.num_rollouts,
            use_synthetic=False,
            device=args.device,
            headless=headless,
            num_traj_wp=args.num_traj_wp,
            max_steps=args.max_steps,
            execution_horizon=args.execution_horizon
        )

        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print(f"Synthetic Demos: {synthetic_sr:.1%}")
        print(f"Real Demos:      {real_sr:.1%}")
        print(f"Difference:      {(synthetic_sr - real_sr)*100:+.1f}%")
        print("="*60)

    else:
        test_synthetic_demo_on_task(
            model, args.task,
            num_rollouts=args.num_rollouts,
            use_synthetic=True,
            vlm_source=args.vlm_source,
            device=args.device,
            headless=headless,
            num_traj_wp=args.num_traj_wp,
            max_steps=args.max_steps,
            execution_horizon=args.execution_horizon
        )


if __name__ == '__main__':
    main()
