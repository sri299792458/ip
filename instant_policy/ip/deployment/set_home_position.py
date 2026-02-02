#!/usr/bin/env python3
"""
Set or record a UR5e home joint position.

Usage examples:
  # Move to a known home (degrees) and save it
  python -m ip.deployment.set_home_position --robot-ip 10.33.55.90 \
    --joints-deg 0 -90 90 -90 -90 0 --save

  # Record current joints as home
  python -m ip.deployment.set_home_position --robot-ip 10.33.55.90 --save-current

  # Move to a previously saved home
  python -m ip.deployment.set_home_position --robot-ip 10.33.55.90 --load ip/deployment/home_joint.json
"""
import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import rtde_control
    import rtde_receive
except Exception as exc:  # pragma: no cover - optional dependency
    rtde_control = None
    rtde_receive = None
    _RTDE_IMPORT_ERROR = exc
else:
    _RTDE_IMPORT_ERROR = None


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


def _require_rtde():
    if rtde_control is None or rtde_receive is None:
        raise ImportError(f"ur_rtde is required: {_RTDE_IMPORT_ERROR}")


def _deg_to_rad(vals: Sequence[float]) -> np.ndarray:
    return np.deg2rad(np.array(vals, dtype=np.float64))


def _rad_to_deg(vals: Sequence[float]) -> np.ndarray:
    return np.rad2deg(np.array(vals, dtype=np.float64))


def _validate_joints(vals: Sequence[float]) -> np.ndarray:
    if len(vals) != 6:
        raise ValueError("Expected 6 joint values")
    return np.array(vals, dtype=np.float64)


def _load_joints(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "joints_rad" in data:
        return _validate_joints(data["joints_rad"])
    if "joints_deg" in data:
        return _deg_to_rad(_validate_joints(data["joints_deg"]))
    raise ValueError("JSON must contain joints_rad or joints_deg")


def _save_joints(path: Path, joints_rad: np.ndarray) -> None:
    payload = {
        "joint_names": JOINT_NAMES,
        "joints_rad": joints_rad.tolist(),
        "joints_deg": _rad_to_deg(joints_rad).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _confirm_or_exit(joints_rad: np.ndarray, require_confirm: bool) -> None:
    if not require_confirm:
        return
    joints_deg = _rad_to_deg(joints_rad)
    print("Target home joints (deg):", np.round(joints_deg, 3).tolist())
    resp = input("Type 'y' to move: ").strip().lower()
    if resp != "y":
        raise SystemExit("Aborted.")


def _wait_until_reached(rx: "rtde_receive.RTDEReceiveInterface",
                        target: np.ndarray,
                        tol_deg: float,
                        max_wait_s: float) -> None:
    import time

    tol_rad = np.deg2rad(tol_deg)
    start = time.time()
    while True:
        current = np.array(rx.getActualQ(), dtype=np.float64)
        err = np.max(np.abs(current - target))
        if err <= tol_rad:
            return
        if time.time() - start > max_wait_s:
            print("Warning: timeout waiting for joints to reach target.")
            return
        time.sleep(0.05)


def main() -> None:
    _require_rtde()
    parser = argparse.ArgumentParser(description="Set or record a UR5e home joint position.")
    parser.add_argument("--robot-ip", required=True, help="Robot IP address")
    parser.add_argument("--joints-deg", type=float, nargs=6, help="Target joints in degrees (6 values)")
    parser.add_argument("--joints-rad", type=float, nargs=6, help="Target joints in radians (6 values)")
    parser.add_argument("--load", type=str, help="Load joints from JSON (joints_rad or joints_deg)")
    parser.add_argument("--save-current", action="store_true", help="Save current joints to --out and exit")
    parser.add_argument(
        "--out",
        type=str,
        default="ip/deployment/home_joint.json",
        help="Output JSON path for --save or --save-current",
    )
    parser.add_argument("--save", action="store_true", help="Save target joints to --out")
    parser.add_argument("--speed", type=float, default=1.0, help="Joint speed (rad/s)")
    parser.add_argument("--accel", type=float, default=1.2, help="Joint acceleration (rad/s^2)")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--wait", action="store_true", help="Wait until target is reached")
    parser.add_argument("--tolerance-deg", type=float, default=1.0, help="Tolerance for --wait in degrees")
    parser.add_argument("--max-wait-s", type=float, default=10.0, help="Max wait time for --wait")
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.save_current:
        rx = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        joints_rad = _validate_joints(rx.getActualQ())
        _save_joints(out_path, joints_rad)
        print(f"Saved current joints to {out_path}")
        return

    target_rad: Optional[np.ndarray] = None
    if args.joints_deg is not None:
        target_rad = _deg_to_rad(_validate_joints(args.joints_deg))
    elif args.joints_rad is not None:
        target_rad = _validate_joints(args.joints_rad)
    elif args.load:
        target_rad = _load_joints(Path(args.load))

    if target_rad is None:
        raise SystemExit("Provide --joints-deg, --joints-rad, --load, or --save-current.")

    if args.save:
        _save_joints(out_path, target_rad)
        print(f"Saved target joints to {out_path}")

    _confirm_or_exit(target_rad, require_confirm=not args.no_confirm)

    rtde = rtde_control.RTDEControlInterface(args.robot_ip)
    rx = None
    try:
        rtde.moveJ(target_rad.tolist(), args.speed, args.accel)
        if args.wait:
            rx = rtde_receive.RTDEReceiveInterface(args.robot_ip)
            _wait_until_reached(rx, target_rad, args.tolerance_deg, args.max_wait_s)
        print("Move complete.")
    finally:
        # Ensure we release remote control so Freedrive can be used from pendant.
        try:
            rtde.stopScript()
        except Exception:
            pass
        try:
            rtde.disconnect()
        except Exception:
            pass
        if rx is not None:
            try:
                rx.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
