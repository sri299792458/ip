"""
Example deployment entrypoint for Instant Policy on UR5e (RTDE, ROS-free).
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

try:
    import rtde_control
except Exception as exc:  # pragma: no cover - optional dependency
    rtde_control = None
    _RTDE_IMPORT_ERROR = exc
else:
    _RTDE_IMPORT_ERROR = None

from ip.deployment.config import CameraConfig, DeploymentConfig
from ip.deployment.control.action_executor import SafetyLimits
from ip.deployment.demo.demo_collector import DemoCollector
from ip.deployment.manual_seed_xmem import manual_seed_xmem
from ip.deployment.orchestrator import InstantPolicyDeployment


def _build_default_config() -> DeploymentConfig:
    config = DeploymentConfig(camera_configs=[])
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


def _require_rtde():
    if rtde_control is None:
        raise ImportError(f"ur_rtde is required: {_RTDE_IMPORT_ERROR}")


def _apply_calibration_json(config: DeploymentConfig, calib_path: Path) -> None:
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    cameras = calib.get("cameras", {})
    if not cameras:
        raise ValueError(f"No cameras found in calibration file: {calib_path}")

    existing_by_serial = {cfg.serial: cfg for cfg in config.camera_configs}
    new_configs = []
    for serial, cam in cameras.items():
        T = np.array(cam.get("T_world_camera"), dtype=np.float64)
        if T.shape != (4, 4):
            continue
        if serial in existing_by_serial:
            base = existing_by_serial[serial]
            new_configs.append(
                CameraConfig(
                    serial=serial,
                    T_world_camera=T,
                    width=base.width,
                    height=base.height,
                    fps=base.fps,
                    align_to_color=base.align_to_color,
                )
            )
        else:
            new_configs.append(CameraConfig(serial=serial, T_world_camera=T))
    if not new_configs:
        raise ValueError(f"No valid T_world_camera entries in calibration file: {calib_path}")
    config.camera_configs = new_configs


def _load_home_joints(args) -> np.ndarray:
    if args.home_joints_deg is not None:
        if len(args.home_joints_deg) != 6:
            raise ValueError("--home-joints-deg expects 6 values")
        return np.deg2rad(np.array(args.home_joints_deg, dtype=np.float64))
    if args.home_joints_rad is not None:
        if len(args.home_joints_rad) != 6:
            raise ValueError("--home-joints-rad expects 6 values")
        return np.array(args.home_joints_rad, dtype=np.float64)
    if args.home:
        with open(args.home, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "joints_rad" in data:
            return np.array(data["joints_rad"], dtype=np.float64)
        if "joints_deg" in data:
            return np.deg2rad(np.array(data["joints_deg"], dtype=np.float64))
        raise ValueError("--home JSON must contain joints_rad or joints_deg")
    raise ValueError("Provide --home, --home-joints-deg, or --home-joints-rad")


def _go_home(args, robot_ip: str) -> None:
    _require_rtde()
    joints_rad = _load_home_joints(args)
    rtde = rtde_control.RTDEControlInterface(robot_ip)
    try:
        rtde.moveJ(joints_rad.tolist(), args.home_speed, args.home_accel)
    finally:
        # Release remote control so pendant Freedrive is available if needed.
        try:
            rtde.stopScript()
        except Exception:
            pass
        try:
            rtde.disconnect()
        except Exception:
            pass


def _open_gripper(control, config: DeploymentConfig) -> None:
    if not config.gripper.enable:
        return
    if hasattr(control, "execute_gripper"):
        control.execute_gripper(1.0)


def main():
    parser = argparse.ArgumentParser(description="Instant Policy deployment on UR5e (RTDE)")
    parser.add_argument("--robot-ip", default=None, help="UR5e IP address (default: config.robot_ip)")
    parser.add_argument(
        "--demo",
        action="append",
        nargs="+",
        default=[],
        metavar="DEMO",
        help=(
            "One or more demo .pkl paths. Can be repeated. "
            "Tip: you can use shell globs, e.g. --demo demos/task1_demo*.pkl"
        ),
    )
    parser.add_argument("--collect-demo", action="store_true", help="Collect a kinesthetic demo and exit")
    parser.add_argument("--demo-out", default="demo.pkl", help="Output path for collected demo")
    parser.add_argument("--task-name", default="task", help="Task name for demo collection")
    parser.add_argument(
        "--debug-demo-waypoints",
        action="store_true",
        help="Save RGB images for the selected waypoint frames during demo collection.",
    )
    parser.add_argument(
        "--debug-demo-waypoints-dir",
        default="ip/deployment/debug_waypoints",
        help="Output dir for waypoint debug images (used with --debug-demo-waypoints).",
    )
    parser.add_argument(
        "--debug-demo-waypoints-num",
        type=int,
        default=None,
        help="Number of waypoint frames to export (default: config.num_traj_wp).",
    )
    parser.add_argument(
        "--camera-serial",
        action="append",
        default=None,
        help="Use only these camera serials (repeatable).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Max execution steps")
    parser.add_argument("--manual-seed", action="store_true", help="Manually seed XMem masks before running")
    parser.add_argument("--manual-seed-out", default=None, help="Optional output dir for saved manual masks")
    parser.add_argument(
        "--calib",
        default="ip/deployment/calibration_outputs/realsense_T_world_camera.json",
        help="Calibration JSON to auto-load (default: left arm calibration)",
    )
    parser.add_argument(
        "--horizon-mode",
        choices=["until-grip-change", "full"],
        default="until-grip-change",
        help=(
            "How many predicted actions to execute per control step: "
            "'until-grip-change' executes until the predicted gripper state flips (recommended), "
            "'full' always executes the full prediction horizon."
        ),
    )
    parser.add_argument(
        "--no-home",
        action="store_false",
        dest="go_home",
        help="Skip moving robot to home joints before starting",
    )
    parser.add_argument("--home", default="ip/deployment/home_joint.json", help="Path to home joint JSON")
    parser.add_argument("--home-joints-deg", type=float, nargs=6, help="Home joints in degrees (6 values)")
    parser.add_argument("--home-joints-rad", type=float, nargs=6, help="Home joints in radians (6 values)")
    parser.add_argument("--home-speed", type=float, default=1.0, help="Home move joint speed (rad/s)")
    parser.add_argument("--home-accel", type=float, default=1.2, help="Home move joint accel (rad/s^2)")
    parser.add_argument(
        "--no-open-gripper",
        action="store_false",
        dest="open_gripper",
        help="Skip opening the gripper before starting",
    )
    parser.add_argument(
        "--viz",
        choices=["none", "masks", "pcd", "both"],
        default="none",
        help="Debug visualization: 'masks', 'pcd' (policy-frame), or 'both'.",
    )
    parser.add_argument(
        "--viz-hz",
        type=float,
        default=None,
        help="Live PCD update rate (Hz) for the viewer (default: config.show_live_pcd_hz).",
    )
    parser.add_argument(
        "--record-live-pcd",
        action="store_true",
        help="Record live policy-frame point clouds to a .pkl (uses config defaults for path/stride).",
    )
    parser.add_argument(
        "--frame",
        choices=["flange", "tip"],
        default="flange",
        help=(
            "End-effector frame convention: "
            "'flange' uses the RTDE-reported TCP pose as-is, "
            "'tip' applies --tcp-offset-m in code. "
            "Default: flange."
        ),
    )
    parser.add_argument(
        "--tcp-offset-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="TCP offset in meters (overrides config when provided).",
    )
    parser.add_argument(
        "--debug-gripper",
        action="store_true",
        help="Print gripper-related debug info from the policy and executor.",
    )
    parser.add_argument(
        "--debug-frame-sanity",
        action="store_true",
        help="Print TCP offset/frame sanity info (tip vs flange) during deployment.",
    )
    parser.add_argument(
        "--debug-frame-every",
        type=int,
        default=1,
        help="Print frame sanity every N steps (default: 1, only used with --debug-frame-sanity).",
    )
    parser.set_defaults(open_gripper=True)
    parser.set_defaults(go_home=True)
    args = parser.parse_args()

    # Flatten --demo which is a list of lists due to nargs="+".
    demo_paths = []
    for item in args.demo:
        if isinstance(item, (list, tuple)):
            demo_paths.extend(item)
        else:
            demo_paths.append(item)
    args.demo = demo_paths

    config = _build_default_config()
    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if args.calib:
        _apply_calibration_json(config, Path(args.calib))
    config.execute_until_grip_change = args.horizon_mode == "until-grip-change"
    config.show_masks = args.viz in {"masks", "both"}
    config.show_live_pcd = args.viz in {"pcd", "both"}
    if args.viz_hz is not None:
        config.show_live_pcd_hz = float(args.viz_hz)
    config.record_live_pcd = args.record_live_pcd
    config.tcp_offset_in_code = args.frame == "tip"
    if args.tcp_offset_m is not None:
        config.tcp_offset_m = np.array(args.tcp_offset_m, dtype=np.float64)
    config.debug_frame_sanity = args.debug_frame_sanity
    config.debug_frame_every = max(1, int(args.debug_frame_every))
    if args.camera_serial:
        serials = set(args.camera_serial)
        filtered = [cfg for cfg in config.camera_configs if cfg.serial in serials]
        if not filtered:
            raise ValueError(
                "Requested camera serials not found in deployment.py config. "
                f"Available: {[cfg.serial for cfg in config.camera_configs]}"
            )
        config.camera_configs = filtered
    if args.calib:
        print(f"Loaded calibration: {args.calib}")
        print(f"Camera serials: {[cfg.serial for cfg in config.camera_configs]}")
    if args.manual_seed:
        config.segmentation.xmem_init_with_sam = False
        if config.segmentation.backend.lower() != "xmem":
            raise ValueError("--manual-seed requires segmentation.backend == 'xmem'")
    frame_label = "TIP" if config.tcp_offset_in_code else "FLANGE"
    print(f"FRAME = {frame_label}")
    if any(cfg.serial.startswith("CAMERA_SERIAL") for cfg in config.camera_configs):
        raise ValueError("Please update camera serials and T_world_camera in deployment.py")

    if args.collect_demo:
        if args.go_home:
            _go_home(args, config.robot_ip)
        deployment = InstantPolicyDeployment(config, load_model=False, debug_gripper=args.debug_gripper)
        if args.open_gripper:
            _open_gripper(deployment.control, config)
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
            debug_waypoints=args.debug_demo_waypoints,
        )
        if args.debug_demo_waypoints:
            num_wp = args.debug_demo_waypoints_num or config.num_traj_wp
            collector.save_waypoint_debug_images(
                raw_demo,
                args.debug_demo_waypoints_dir,
                num_traj_wp=num_wp,
            )
            raw_demo.pop("_debug_rgb", None)
        collector.save_demo(raw_demo, args.demo_out)
        print(f"Saved demo to {args.demo_out}")
        return

    if args.go_home:
        _go_home(args, config.robot_ip)
    deployment = InstantPolicyDeployment(config, debug_gripper=args.debug_gripper)
    if args.open_gripper:
        _open_gripper(deployment.control, config)
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
