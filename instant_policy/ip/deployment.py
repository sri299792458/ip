"""
Example deployment entrypoint for Instant Policy on UR5e (RTDE, ROS-free).
"""
import argparse
import pickle

import numpy as np

from ip.deployment.config import CameraConfig, DeploymentConfig
from ip.deployment.control.action_executor import SafetyLimits
from ip.deployment.demo.demo_collector import DemoCollector
from ip.deployment.manual_seed_xmem import manual_seed_xmem
from ip.deployment.orchestrator import InstantPolicyDeployment


def _build_default_config() -> DeploymentConfig:
    camera_configs = [
        CameraConfig(
            serial="f1380660",
            T_world_camera=np.array(
                [
                    [0.9952, 0.0067, -0.0978, -0.4402],
                    [-0.0729, -0.6154, -0.7849, 1.1759],
                    [-0.0654, 0.7882, -0.6119, 0.0865],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        CameraConfig(
            serial="f1371463",
            T_world_camera=np.array(
                [
                    [0.9978, 0.0556, -0.0352, -0.5160],
                    [-0.0283, -0.1209, -0.9923, 1.5065],
                    [-0.0595, 0.9911, -0.1190, -0.0123],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
    ]
    config = DeploymentConfig(camera_configs=camera_configs)
    config.robot_ip = "10.33.55.90"
    config.model_path = "./checkpoints/ip"
    config.segmentation.backend = "xmem"
    config.segmentation.sam_checkpoint_path = "./checkpoints/sam/sam_vit_b_01ec64.pth"
    config.segmentation.xmem_checkpoint_path = "./checkpoints/xmem/XMem.pth"
    config.segmentation.enable = True
    config.device = "cuda:0"
    config.rtde.move_speed = 0.05
    config.rtde.move_acceleration = 0.2
    config.safety = SafetyLimits(
        workspace_min=np.array([-0.9008, 0.2936, -0.4819]),
        workspace_max=np.array([-0.1227, 0.5751, 0.5293]),
        max_translation=0.01,
        max_rotation=np.deg2rad(3.0),
    )
    return config


def _load_demos(paths):
    demos = []
    for path in paths:
        with open(path, "rb") as f:
            demos.append(pickle.load(f))
    return demos


def main():
    parser = argparse.ArgumentParser(description="Instant Policy deployment on UR5e (RTDE)")
    parser.add_argument("--robot-ip", default=None, help="UR5e IP address (default: config.robot_ip)")
    parser.add_argument("--demo", action="append", default=[], help="Path to a demo pickle file")
    parser.add_argument("--collect-demo", action="store_true", help="Collect a kinesthetic demo and exit")
    parser.add_argument("--demo-out", default="demo.pkl", help="Output path for collected demo")
    parser.add_argument("--task-name", default="task", help="Task name for demo collection")
    parser.add_argument("--max-steps", type=int, default=None, help="Max execution steps")
    parser.add_argument("--manual-seed", action="store_true", help="Manually seed XMem masks before running")
    parser.add_argument("--manual-seed-out", default=None, help="Optional output dir for saved manual masks")
    args = parser.parse_args()

    config = _build_default_config()
    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if args.manual_seed:
        config.segmentation.xmem_init_with_sam = False
        if config.segmentation.backend.lower() != "xmem":
            raise ValueError("--manual-seed requires segmentation.backend == 'xmem'")
    if any(cfg.serial.startswith("CAMERA_SERIAL") for cfg in config.camera_configs):
        raise ValueError("Please update camera serials and T_world_camera in deployment.py")

    if args.collect_demo:
        deployment = InstantPolicyDeployment(config, load_model=False)
        if args.manual_seed and config.segmentation.enable:
            manual_seed_xmem(
                deployment.perception,
                [cfg.serial for cfg in config.camera_configs],
                out_dir=args.manual_seed_out,
            )
        collector = DemoCollector(deployment.perception, deployment.state, deployment.control)
        raw_demo = collector.collect_kinesthetic(
            args.task_name,
            use_segmentation=config.segmentation.enable,
        )
        collector.save_demo(raw_demo, args.demo_out)
        print(f"Saved demo to {args.demo_out}")
        return

    deployment = InstantPolicyDeployment(config)
    if args.manual_seed and config.segmentation.enable:
        manual_seed_xmem(
            deployment.perception,
            [cfg.serial for cfg in config.camera_configs],
            out_dir=args.manual_seed_out,
        )
    demos = _load_demos(args.demo)
    deployment.run(demos, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
