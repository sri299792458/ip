#!/usr/bin/env python3
"""
Replay a recorded demo trajectory on hardware.

Assumes demo contains T_w_es in the same frame you intend to command
(default: flange frame, tcp_offset_in_code=False).
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

try:
    import rtde_control
except Exception as exc:  # pragma: no cover - optional dependency
    rtde_control = None
    _RTDE_CONTROL_IMPORT_ERROR = exc
else:
    _RTDE_CONTROL_IMPORT_ERROR = None

try:
    import rtde_receive
except Exception as exc:  # pragma: no cover - optional dependency
    rtde_receive = None
    _RTDE_RECEIVE_IMPORT_ERROR = exc
else:
    _RTDE_RECEIVE_IMPORT_ERROR = None

from ip.deployment.config import GripperConfig, RTDEControlConfig
from ip.deployment.control.action_executor import ActionExecutor, SafetyLimits
from ip.deployment.control.ur_rtde_control import URRTDEControl
from ip.deployment.state.ur_rtde_state import URRTDEState
from ip.deployment.ur.robotiq_gripper import RobotiqGripper


def _require_rtde():
    if rtde_control is None:
        raise ImportError(f"ur_rtde (control) is required: {_RTDE_CONTROL_IMPORT_ERROR}")
    if rtde_receive is None:
        raise ImportError(f"ur_rtde (receive) is required: {_RTDE_RECEIVE_IMPORT_ERROR}")


def _prompt(msg: str) -> bool:
    resp = input(f"{msg} [y/N] ").strip().lower()
    return resp in {"y", "yes"}


def _load_demo(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _indices(n: int, start: int, end: int, stride: int):
    start = max(0, start)
    end = n - 1 if end < 0 else min(end, n - 1)
    return list(range(start, end + 1, max(1, stride)))


def main():
    _require_rtde()
    parser = argparse.ArgumentParser(description="Replay a demo trajectory on hardware.")
    parser.add_argument("--demo", required=True, help="Path to demo .pkl")
    parser.add_argument("--robot-ip", required=True, help="Robot IP address")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (inclusive, -1 for last)")
    parser.add_argument("--stride", type=int, default=1, help="Play every Nth frame")
    parser.add_argument("--go-start", action="store_true", help="Move to first pose before replay")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--dry-run", action="store_true", help="Print actions but do not move")
    parser.add_argument("--hold-sec", type=float, default=0.0, help="Sleep after each step (seconds)")
    parser.add_argument("--control-mode", choices=["moveL", "servoL"], default="moveL")
    parser.add_argument("--move-speed", type=float, default=0.1, help="moveL speed (m/s)")
    parser.add_argument("--move-accel", type=float, default=0.5, help="moveL accel (m/s^2)")
    parser.add_argument("--servo-speed", type=float, default=0.1, help="servoL speed (m/s)")
    parser.add_argument("--servo-accel", type=float, default=0.5, help="servoL accel (m/s^2)")
    parser.add_argument("--servo-time", type=float, default=0.1, help="servoL time (s)")
    parser.add_argument("--servo-lookahead", type=float, default=0.1, help="servoL lookahead (s)")
    parser.add_argument("--servo-gain", type=int, default=300, help="servoL gain")
    parser.add_argument("--max-translation", type=float, default=0.01, help="Max translation per step (m)")
    parser.add_argument("--max-rotation-deg", type=float, default=3.0, help="Max rotation per step (deg)")
    parser.add_argument(
        "--tcp-offset-in-code",
        action="store_true",
        dest="tcp_offset_in_code",
        help="Apply TCP offset in code (tip frame).",
    )
    parser.add_argument(
        "--no-tcp-offset-in-code",
        action="store_false",
        dest="tcp_offset_in_code",
        help="Do not apply TCP offset in code (flange frame).",
    )
    parser.add_argument(
        "--tcp-offset-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="TCP offset in meters (default: 0 0 0.162 if --tcp-offset-in-code).",
    )
    parser.add_argument("--use-gripper", action="store_true", help="Replay gripper commands from demo")
    parser.add_argument("--debug-gripper", action="store_true", help="Print gripper commands during replay")
    parser.add_argument("--gripper-host", default=None, help="Robotiq gripper host (default: robot-ip)")
    parser.add_argument("--gripper-port", type=int, default=63352)
    parser.add_argument("--gripper-open", type=int, default=0)
    parser.add_argument("--gripper-closed", type=int, default=255)
    parser.add_argument("--gripper-speed", type=int, default=255)
    parser.add_argument("--gripper-force", type=int, default=100)
    parser.set_defaults(tcp_offset_in_code=False)
    args = parser.parse_args()

    demo = _load_demo(Path(args.demo))
    T_w_es = demo.get("T_w_es", [])
    grips = demo.get("grips", [])
    if not T_w_es:
        raise RuntimeError("Demo has no T_w_es.")
    if args.use_gripper and not grips:
        raise RuntimeError("Demo has no grips; re-collect with gripper states or omit --use-gripper.")
    idxs = _indices(len(T_w_es), args.start, args.end, args.stride)
    if not idxs:
        raise RuntimeError("No frames in selected range.")

    tcp_offset = None
    if args.tcp_offset_in_code:
        tcp_offset = np.array([0.0, 0.0, 0.162], dtype=np.float64)
        if args.tcp_offset_m is not None:
            tcp_offset = np.array(args.tcp_offset_m, dtype=np.float64)

    gripper = None
    if args.use_gripper:
        host = args.gripper_host or args.robot_ip
        gripper = RobotiqGripper(
            host=host,
            port=args.gripper_port,
            open_position=args.gripper_open,
            closed_position=args.gripper_closed,
        )
        gripper.connect()
        gripper.activate()
        if args.debug_gripper:
            sample_vals = grips[: min(10, len(grips))]
            print(f"Gripper demo values (first {len(sample_vals)}): {sample_vals}")

    rtde_cfg = RTDEControlConfig(
        control_mode=args.control_mode,
        move_speed=args.move_speed,
        move_acceleration=args.move_accel,
        servo_speed=args.servo_speed,
        servo_acceleration=args.servo_accel,
        servo_time=args.servo_time,
        servo_lookahead=args.servo_lookahead,
        servo_gain=args.servo_gain,
    )
    rtde_control = URRTDEControl.connect(args.robot_ip, rtde_cfg)
    rtde_receive_iface = URRTDEState.connect(args.robot_ip)
    control = URRTDEControl(
        rtde_control,
        rtde_cfg,
        gripper=gripper,
        gripper_config=GripperConfig(
            enable=args.use_gripper,
            host=args.gripper_host or args.robot_ip,
            port=args.gripper_port,
            open_position=args.gripper_open,
            closed_position=args.gripper_closed,
            speed=args.gripper_speed,
            force=args.gripper_force,
        ),
        tcp_offset_in_code=args.tcp_offset_in_code,
        tcp_offset_m=tcp_offset,
    )
    state = URRTDEState(
        rtde_receive_iface,
        gripper=gripper,
        tcp_offset_in_code=args.tcp_offset_in_code,
        tcp_offset_m=tcp_offset,
    )
    safety = SafetyLimits(
        max_translation=args.max_translation,
        max_rotation=np.deg2rad(args.max_rotation_deg),
    )
    executor = ActionExecutor(control, state, safety=safety, debug_gripper=False)

    frame_label = "TIP" if args.tcp_offset_in_code else "FLANGE"
    print(f"FRAME = {frame_label}")
    print(f"Replaying {len(idxs)} frames (stride={args.stride}).")

    if args.go_start:
        first_idx = idxs[0]
        T_start = np.asarray(T_w_es[first_idx], dtype=np.float64)
        if not args.no_confirm:
            if not _prompt(f"Move to start frame {first_idx}?"):
                return
        if args.dry_run:
            print("Dry run: would move to start pose.")
        else:
            control.execute_pose(T_start)

    if not args.no_confirm:
        if not _prompt("Begin replay?"):
            return

    for i, idx in enumerate(idxs):
        T_target = np.asarray(T_w_es[idx], dtype=np.float64)
        if T_target.shape != (4, 4):
            print(f"Skipping frame {idx}: invalid T_w_e shape {T_target.shape}")
            continue
        grip_val = None
        if args.use_gripper and idx < len(grips):
            grip_val = 1.0 if grips[idx] >= 0.5 else 0.0
            if args.debug_gripper:
                state_label = "open" if grip_val >= 0.5 else "close"
                print(f"[gripper] frame {idx}: demo={grips[idx]} -> {state_label}")
        if args.dry_run:
            print(f"[{i+1}/{len(idxs)}] frame {idx} (dry run)")
            continue

        T_current = state.get_T_w_e()
        T_rel = np.linalg.inv(T_current) @ T_target
        actions = np.expand_dims(T_rel, axis=0)
        if grip_val is None:
            grips_rel = np.array([1.0], dtype=np.float64)
        else:
            grips_rel = np.array([1.0 if grip_val >= 0.5 else -1.0], dtype=np.float64)
        ok, steps, reason = executor.execute_actions(actions, grips_rel, T_current, horizon=1)
        if not ok:
            print(f"Stopped at frame {idx}: {reason}")
            break
        if args.hold_sec > 0:
            time.sleep(args.hold_sec)


if __name__ == "__main__":
    main()
