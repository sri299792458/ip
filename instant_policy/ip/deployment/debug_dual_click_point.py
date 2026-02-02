#!/usr/bin/env python3
"""
Click the same physical point in two camera views and compare world points.
Requires a calibration JSON with T_world_camera for both serials.
"""
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

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
    T = np.array(cams[serial]["T_world_camera"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_world_camera for {serial} is not 4x4")
    return T


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


def _start_pipeline(serial: str, width: int, height: int, fps: int):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    return pipeline, align, intr, depth_scale


def _draw_click_marker(bgr: np.ndarray, pixel: Optional[Tuple[int, int]]) -> np.ndarray:
    if pixel is None:
        return bgr
    x, y = pixel
    out = bgr.copy()
    cv2.drawMarker(out, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    return out


def main():
    _require_deps()
    parser = argparse.ArgumentParser(
        description="Click the same point in two RealSense views and compare world points."
    )
    parser.add_argument("--serial-a", required=True, help="RealSense serial for camera A")
    parser.add_argument("--serial-b", required=True, help="RealSense serial for camera B")
    parser.add_argument(
        "--calib",
        default=str(Path(__file__).resolve().parent / "calibration_outputs" / "realsense_T_world_camera.json"),
        help="Path to calibration JSON",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Warmup frames before showing windows.",
    )
    args = parser.parse_args()

    T_world_a = _load_T_world_camera(Path(args.calib), args.serial_a)
    T_world_b = _load_T_world_camera(Path(args.calib), args.serial_b)

    pipe_a, align_a, intr_a, scale_a = _start_pipeline(args.serial_a, args.width, args.height, args.fps)
    pipe_b, align_b, intr_b, scale_b = _start_pipeline(args.serial_b, args.width, args.height, args.fps)

    last_depth_a = None
    last_depth_b = None
    last_pix_a = None
    last_pix_b = None
    last_world_a = None
    last_world_b = None
    world_pairs = []

    def reset_clicks():
        nonlocal last_pix_a, last_pix_b, last_world_a, last_world_b
        last_pix_a = None
        last_pix_b = None
        last_world_a = None
        last_world_b = None
        print("Cleared last clicks.")

    def clear_pairs():
        nonlocal world_pairs
        world_pairs = []
        print("Cleared all stored point pairs.")

    def _maybe_compare():
        if last_world_a is None or last_world_b is None:
            return
        delta = last_world_a - last_world_b
        print("\n=== World point comparison ===")
        print(f"A world: {np.round(last_world_a, 4)}")
        print(f"B world: {np.round(last_world_b, 4)}")
        print(f"Delta (A - B): {np.round(delta, 4)} (norm {np.linalg.norm(delta):.4f} m)")
        print("==============================\n")

    def _store_pair():
        if last_world_a is None or last_world_b is None:
            print("Need both A and B clicks before storing a pair.")
            return
        world_pairs.append((last_world_a.copy(), last_world_b.copy()))
        print(f"Stored pair #{len(world_pairs)}.")

    def _solve_rigid():
        if len(world_pairs) < 3:
            print("Need at least 3 point pairs to solve a rigid transform.")
            return
        A = np.stack([p[0] for p in world_pairs], axis=0)
        B = np.stack([p[1] for p in world_pairs], axis=0)
        cA = A.mean(axis=0)
        cB = B.mean(axis=0)
        AA = A - cA
        BB = B - cB
        H = BB.T @ AA
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = cA - R @ cB
        A_hat = (R @ B.T).T + t
        errs = np.linalg.norm(A_hat - A, axis=1)
        rmse = float(np.sqrt(np.mean(errs**2)))
        mean_delta = (A - B).mean(axis=0)
        print("\n=== Best-fit B->A rigid transform ===")
        print("R (B->A):")
        print(np.round(R, 6))
        print("t (B->A):", np.round(t, 6).tolist())
        print(f"RMSE: {rmse:.6f} m")
        print("Mean (A - B):", np.round(mean_delta, 6).tolist())
        print("=====================================\n")

    def on_mouse_a(event, x, y, flags, param):
        nonlocal last_depth_a, last_pix_a, last_world_a
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if last_depth_a is None:
            print("A: no depth frame yet.")
            return
        d = _depth_at(last_depth_a, x, y)
        if d is None:
            print(f"A: pixel ({x},{y}) no valid depth")
            return
        p_cam = _pixel_to_cam(x, y, d, intr_a)
        p_world = (T_world_a[:3, :3] @ p_cam) + T_world_a[:3, 3]
        last_pix_a = (x, y)
        last_world_a = p_world
        print(f"A: pixel ({x},{y}) depth {d:.4f} m, cam {np.round(p_cam, 4)}, world {np.round(p_world, 4)}")
        _maybe_compare()

    def on_mouse_b(event, x, y, flags, param):
        nonlocal last_depth_b, last_pix_b, last_world_b
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if last_depth_b is None:
            print("B: no depth frame yet.")
            return
        d = _depth_at(last_depth_b, x, y)
        if d is None:
            print(f"B: pixel ({x},{y}) no valid depth")
            return
        p_cam = _pixel_to_cam(x, y, d, intr_b)
        p_world = (T_world_b[:3, :3] @ p_cam) + T_world_b[:3, 3]
        last_pix_b = (x, y)
        last_world_b = p_world
        print(f"B: pixel ({x},{y}) depth {d:.4f} m, cam {np.round(p_cam, 4)}, world {np.round(p_world, 4)}")
        _maybe_compare()

    for _ in range(args.warmup_frames):
        pipe_a.wait_for_frames()
        pipe_b.wait_for_frames()

    win_a = f"camA {args.serial_a}"
    win_b = f"camB {args.serial_b}"
    cv2.namedWindow(win_a, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_b, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_a, on_mouse_a)
    cv2.setMouseCallback(win_b, on_mouse_b)
    print(
        "Click the same physical point in both windows. "
        "Keys: 'p' store pair, 's' solve, 'r' reset clicks, 'c' clear pairs, 'q' quit."
    )

    try:
        while True:
            frames_a = pipe_a.wait_for_frames()
            frames_b = pipe_b.wait_for_frames()
            if align_a is not None:
                frames_a = align_a.process(frames_a)
            if align_b is not None:
                frames_b = align_b.process(frames_b)

            depth_a = frames_a.get_depth_frame()
            color_a = frames_a.get_color_frame()
            depth_b = frames_b.get_depth_frame()
            color_b = frames_b.get_color_frame()
            if not depth_a or not color_a or not depth_b or not color_b:
                continue

            last_depth_a = np.asanyarray(depth_a.get_data()).astype(np.float32) * scale_a
            last_depth_b = np.asanyarray(depth_b.get_data()).astype(np.float32) * scale_b
            bgr_a = np.asanyarray(color_a.get_data())
            bgr_b = np.asanyarray(color_b.get_data())

            bgr_a = _draw_click_marker(bgr_a, last_pix_a)
            bgr_b = _draw_click_marker(bgr_b, last_pix_b)

            cv2.imshow(win_a, bgr_a)
            cv2.imshow(win_b, bgr_b)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                reset_clicks()
            if key == ord("c"):
                clear_pairs()
            if key == ord("p"):
                _store_pair()
            if key == ord("s"):
                _solve_rigid()
    finally:
        try:
            pipe_a.stop()
        except Exception:
            pass
        try:
            pipe_b.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
