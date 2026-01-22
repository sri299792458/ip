"""
Example deployment entrypoint for Instant Policy on UR5e (RTDE, ROS-free).
"""
import argparse
import pickle

import numpy as np

from ip.deployment.config import CameraConfig, DeploymentConfig
from ip.deployment.demo.demo_collector import DemoCollector
from ip.deployment.orchestrator import InstantPolicyDeployment


def _build_default_config() -> DeploymentConfig:
    camera_configs = [
        CameraConfig(serial="CAMERA_SERIAL_1", T_world_camera=np.eye(4)),
        CameraConfig(serial="CAMERA_SERIAL_2", T_world_camera=np.eye(4)),
    ]
    config = DeploymentConfig(camera_configs=camera_configs)
    config.segmentation.enable = True
    config.segmentation.backend = "xmem"
    config.segmentation.xmem_checkpoint_path = "../XMem2-main/saves/XMem.pth"
    config.segmentation.sam_checkpoint_path = "../path/to/sam_vit_b.pth"
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
    args = parser.parse_args()

    config = _build_default_config()
    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if any(cfg.serial.startswith("CAMERA_SERIAL") for cfg in config.camera_configs):
        raise ValueError("Please update camera serials and T_world_camera in deployment.py")

    deployment = InstantPolicyDeployment(config)
    collector = DemoCollector(deployment.perception, deployment.state, deployment.control)

    if args.collect_demo:
        raw_demo = collector.collect_kinesthetic(args.task_name)
        collector.save_demo(raw_demo, args.demo_out)
        print(f"Saved demo to {args.demo_out}")
        return

    demos = _load_demos(args.demo)
    deployment.run(demos, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
