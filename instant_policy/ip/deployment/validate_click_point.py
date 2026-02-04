#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - optional dependency
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import pyrealsense2 as rs
except Exception as exc:  # pragma: no cover - optional dependency
    rs = None
    _RS_IMPORT_ERROR = exc
else:
    _RS_IMPORT_ERROR = None

try:
    import rtde_receive
except Exception:  # pragma: no cover - optional dependency
    rtde_receive = None

from ip.deployment.config import RTDEControlConfig


def _require_deps():
    if rs is None:
        raise ImportError(f"pyrealsense2 is required: {_RS_IMPORT_ERROR}")
    if cv2 is None:
        raise ImportError(f"OpenCV is required: {_CV2_IMPORT_ERROR}")


def _load_T_world_camera(calib_path: Path, serial: str) -> np.ndarray:
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with calib_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cams = data.get("cameras", {})
    if serial not in cams:
        raise KeyError(f"Serial {serial} not found in {calib_path}")
    return np.array(cams[serial]["T_world_camera"], dtype=np.float64)


def _pixel_to_cam(u: int, v: int, depth_m: float, intr) -> np.ndarray:
    x = (u - intr.ppx) / intr.fx * depth_m
    y = (v - intr.ppy) / intr.fy * depth_m
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def _depth_at(depth_m: np.ndarray, u: int, v: int) -> Optional[float]:
    h, w = depth_m.shape
    if u < 0 or v < 0 or u >= w or v >= h:
        return None
    d = float(depth_m[v, u])
    if d > 0:
        return d
    # Fallback: median in a small window if center depth is missing.
    u0, u1 = max(0, u - 2), min(w, u + 3)
    v0, v1 = max(0, v - 2), min(h, v + 3)
    window = depth_m[v0:v1, u0:u1].reshape(-1)
    window = window[window > 0]
    if window.size == 0:
        return None
    return float(np.median(window))


def main():
    _require_deps()
    parser = argparse.ArgumentParser(description="Click a pixel and print world point using T_world_camera.")
    parser.add_argument("--serial", required=True, help="RealSense serial to use")
    parser.add_argument(
        "--calib",
        default=str(Path(__file__).resolve().parent / "calibration_outputs" / "realsense_T_world_camera.json"),
        help="Path to calibration JSON",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-ip", default=None, help="Optional: print TCP pose for comparison")
    parser.add_argument(
        "--move-on-click",
        action="store_true",
        help="Enable moving the robot to the last clicked point (press 'm' to execute).",
    )
    parser.add_argument(
        "--auto-move",
        action="store_true",
        help="Move automatically after a click (use with caution).",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt before moving.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["moveL", "servoL"],
        default="moveL",
        help="Motion mode for RTDE control.",
    )
    parser.add_argument("--move-speed", type=float, default=0.1, help="moveL speed (m/s)")
    parser.add_argument("--move-acceleration", type=float, default=0.5, help="moveL acceleration (m/s^2)")
    parser.add_argument(
        "--frame",
        choices=["flange", "tip"],
        default="flange",
        help="End-effector frame convention for interpreting robot TCP pose.",
    )
    parser.add_argument(
        "--tcp-offset-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="TCP offset in meters (only used when --frame tip; default: 0 0 0.162).",
    )
    parser.add_argument(
        "--approach-z-offset",
        type=float,
        default=0.0,
        help="Optional Z offset (meters) to approach above the point before final move.",
    )
    args = parser.parse_args()

    T_world_camera = _load_T_world_camera(Path(args.calib), args.serial)

    rtde = None
    move_enabled = args.move_on_click or args.auto_move
    if move_enabled and not args.robot_ip:
        raise ValueError("--robot-ip is required when --move-on-click or --auto-move is set")

    rtde_state = None
    control = None
    tcp_offset_in_code = args.frame == "tip"
    tcp_offset = None
    if tcp_offset_in_code:
        tcp_offset = np.array([0.0, 0.0, 0.162], dtype=np.float64)
        if args.tcp_offset_m is not None:
            tcp_offset = np.array(args.tcp_offset_m, dtype=np.float64)

    if args.robot_ip:
        if rtde_receive is None:
            raise ImportError("ur_rtde is required for --robot-ip")
        rtde = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    if move_enabled:
        from ip.deployment.control.ur_rtde_control import URRTDEControl
        from ip.deployment.state.ur_rtde_state import URRTDEState

        rtde_cfg = RTDEControlConfig(
            control_mode=args.control_mode,
            move_speed=args.move_speed,
            move_acceleration=args.move_acceleration,
        )
        rtde_control = URRTDEControl.connect(args.robot_ip, rtde_cfg)
        control = URRTDEControl(
            rtde_control,
            rtde_cfg,
            tcp_offset_in_code=tcp_offset_in_code,
            tcp_offset_m=tcp_offset,
        )
        rtde_state = URRTDEState(
            rtde,
            tcp_offset_in_code=tcp_offset_in_code,
            tcp_offset_m=tcp_offset,
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    window = "click-to-world"
    last_depth = None
    last_world = None
    pending_move = False

    def move_to_point(p_world: np.ndarray) -> None:
        if control is None or rtde_state is None:
            print("Robot move requested but RTDE control/state is unavailable.")
            return
        T_w_e_current = rtde_state.get_T_w_e()
        R_current = T_w_e_current[:3, :3]
        T_target = np.eye(4)
        T_target[:3, :3] = R_current
        T_target[:3, 3] = p_world

        if not args.no_confirm:
            confirm = input(f"Move robot to {p_world.round(4)}? [y/N] ").strip().lower()
            if confirm not in {"y", "yes"}:
                print("Move canceled.")
                return

        if args.approach_z_offset > 0:
            p_approach = p_world.copy()
            p_approach[2] += args.approach_z_offset
            T_approach = np.eye(4)
            T_approach[:3, :3] = R_current
            T_approach[:3, 3] = p_approach
            control.execute_pose(T_approach)

        control.execute_pose(T_target)

    def on_mouse(event, x, y, flags, param):
        nonlocal last_depth
        nonlocal last_world
        nonlocal pending_move
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if last_depth is None:
            print("No depth frame yet.")
            return
        d = _depth_at(last_depth, x, y)
        if d is None:
            print(f"Pixel ({x},{y}): no valid depth")
            return
        p_cam = _pixel_to_cam(x, y, d, intr)
        p_world = (T_world_camera[:3, :3] @ p_cam) + T_world_camera[:3, 3]
        print(f"Pixel ({x},{y}) depth {d:.4f} m")
        print(f"Camera point: [{p_cam[0]:.4f}, {p_cam[1]:.4f}, {p_cam[2]:.4f}] m")
        print(f"World point : [{p_world[0]:.4f}, {p_world[1]:.4f}, {p_world[2]:.4f}] m")
        if rtde is not None:
            pose = rtde.getActualTCPPose()
            print(
                "TCP pose   : "
                f"[{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}, "
                f"{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}]"
            )
            delta = np.array(pose[:3]) - p_world
            print(f"Delta (TCP - point): [{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}] m")
        last_world = p_world
        if move_enabled:
            pending_move = True
            if not args.auto_move:
                print("Press 'm' to move robot to this point.")

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            last_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            color = np.asanyarray(color_frame.get_data())
            cv2.imshow(window, color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("m") and move_enabled and last_world is not None:
                pending_move = False
                move_to_point(last_world)
            if args.auto_move and pending_move and last_world is not None:
                pending_move = False
                move_to_point(last_world)
            if key == ord("q") or key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
