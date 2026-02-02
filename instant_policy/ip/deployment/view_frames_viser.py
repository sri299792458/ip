#!/usr/bin/env python3
"""
Visualize world/base and camera frames in Viser using T_world_camera.

Frames:
  - /base: robot base/world frame (origin in world coordinates)
  - /camera/<serial>: camera frame from calibration JSON (T_world_camera)
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import viser
except Exception as exc:  # pragma: no cover - optional dependency
    viser = None
    _VISER_IMPORT_ERROR = exc
else:
    _VISER_IMPORT_ERROR = None


def _require_viser():
    if viser is None:
        raise ImportError(f"viser is required: {_VISER_IMPORT_ERROR}")


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q


def main():
    _require_viser()
    parser = argparse.ArgumentParser(description="Visualize world + camera frames in Viser.")
    parser.add_argument(
        "--calib",
        default=str(Path(__file__).resolve().parent / "calibration_outputs" / "realsense_T_world_camera.json"),
        help="Path to calibration JSON",
    )
    parser.add_argument(
        "--serial",
        action="append",
        required=True,
        help="RealSense serial to visualize (repeat for multiple).",
    )
    parser.add_argument("--axis-length", type=float, default=0.1, help="Axis length (meters)")
    parser.add_argument("--axis-radius", type=float, default=0.005, help="Axis radius (meters)")
    args = parser.parse_args()

    calib_path = Path(args.calib)
    with calib_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cams = data.get("cameras", {})
    serials = args.serial
    for s in serials:
        if s not in cams:
            raise KeyError(f"Serial {s} not found in {calib_path}")

    server = viser.ViserServer()
    server.scene.world_axes.visible = True

    # Base/world frame at origin.
    server.scene.add_frame(
        "/base",
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        axes_length=args.axis_length,
        axes_radius=args.axis_radius,
    )

    # Camera frames using T_world_camera.
    for s in serials:
        T = np.array(cams[s]["T_world_camera"], dtype=np.float64)
        R = T[:3, :3]
        t = T[:3, 3]
        q = _quat_from_matrix(R)
        server.scene.add_frame(
            f"/camera/{s}",
            position=np.asarray(t, dtype=np.float32),
            wxyz=np.asarray(q, dtype=np.float32),
            axes_length=args.axis_length,
            axes_radius=args.axis_radius,
        )

    server.sleep_forever()


if __name__ == "__main__":
    main()
