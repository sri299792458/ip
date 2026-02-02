#!/usr/bin/env python3
"""
Viser debug: overlay demo vs live point clouds in the *world/base frame*.

Why this helps:
- Demo point clouds are recorded in world frame (capture_pcd_world).
- Live point clouds are reconstructed in world frame using T_world_camera.
If these don't align, calibration or segmentation is wrong.
"""
import argparse
import json
import pickle
import threading
import time
from pathlib import Path

import numpy as np

try:
    import viser
except Exception as exc:  # pragma: no cover - optional dependency
    viser = None
    _VISER_IMPORT_ERROR = exc
else:
    _VISER_IMPORT_ERROR = None

from ip.deployment.config import CameraConfig, DeploymentConfig
from ip.deployment.perception.realsense_perception import RealSensePerception
from ip.deployment.perception.sam_segmentation import build_segmenter


def _require_viser():
    if viser is None:
        raise ImportError(f"viser is required: {_VISER_IMPORT_ERROR}")


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx]


def _load_config_defaults() -> DeploymentConfig:
    import importlib.util
    entry = Path(__file__).resolve().parents[1] / "deployment.py"
    spec = importlib.util.spec_from_file_location("ip_deploy_entry", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_default_config()


def _load_calib(calib_path: Path, serials: list[str] | None, width: int, height: int, fps: int):
    with calib_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cams = data.get("cameras", {})
    if not cams:
        raise ValueError(f"No cameras found in calibration file: {calib_path}")
    if not serials:
        serials = list(cams.keys())
    camera_configs = []
    for s in serials:
        if s not in cams:
            raise KeyError(f"Serial {s} not found in {calib_path}")
        T = np.array(cams[s]["T_world_camera"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_world_camera for {s} is not 4x4")
        camera_configs.append(CameraConfig(serial=s, T_world_camera=T, width=width, height=height, fps=fps))
    return camera_configs


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
    parser = argparse.ArgumentParser(description="Overlay demo vs live PCDs in world frame (Viser).")
    parser.add_argument("--demo", required=True, help="Path to demo .pkl (with pcds)")
    parser.add_argument("--frame", type=int, default=0, help="Demo frame index to display")
    parser.add_argument(
        "--calib",
        default=str(Path(__file__).resolve().parent / "calibration_outputs" / "realsense_T_world_camera.json"),
        help="Calibration JSON path",
    )
    parser.add_argument("--serial", action="append", help="Camera serial(s) to use (repeatable)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--show-axes", action="store_true", help="Show world axes")
    parser.add_argument("--show-cameras", action="store_true", help="Show camera frames")
    parser.add_argument("--show-ee", action="store_true", help="Show end-effector frame (requires --ee-from-demo)")
    parser.add_argument(
        "--ee-from-demo",
        type=int,
        default=None,
        help="Demo frame index to use for EE pose (uses T_w_e from demo).",
    )
    parser.add_argument("--show-demo-tcp", action="store_true", help="Show demo TCP for selected frame")
    parser.add_argument("--tcp-radius", type=float, default=0.01)
    parser.add_argument("--demo-max-points", type=int, default=80000)
    parser.add_argument("--live-max-points", type=int, default=80000)
    parser.add_argument("--live-refresh-hz", type=float, default=2.0, help="Live update rate (0 to disable)")
    parser.add_argument("--use-config", action="store_true", help="Load deployment config for segmentation defaults")
    parser.add_argument("--use-segmentation", action="store_true", help="Enable segmentation for live capture")
    parser.add_argument("--manual-seed", action="store_true", help="Manually seed XMem masks (requires --use-config)")
    parser.add_argument("--manual-seed-out", default=None, help="Optional output dir for saved manual masks")
    parser.add_argument("--device", default=None, help="Device for segmentation (e.g., cuda:0)")
    args = parser.parse_args()

    demo_path = Path(args.demo)
    with demo_path.open("rb") as f:
        demo = pickle.load(f)
    pcds = demo.get("pcds", [])
    T_w_es = demo.get("T_w_es", [])
    if not pcds:
        raise RuntimeError("Demo has no point clouds.")
    frame_idx = int(np.clip(args.frame, 0, len(pcds) - 1))
    demo_pts = np.asarray(pcds[frame_idx], dtype=np.float32)
    demo_pts = _subsample(demo_pts, args.demo_max_points)

    calib_path = Path(args.calib)
    camera_configs = _load_calib(calib_path, args.serial, args.width, args.height, args.fps)

    segmenter = None
    voxel_size = None
    if args.manual_seed:
        if not args.use_config:
            raise ValueError("--manual-seed requires --use-config (to load checkpoints).")
        args.use_segmentation = True

    if args.use_config:
        cfg = _load_config_defaults()
        if args.device:
            cfg.device = args.device
        if args.manual_seed:
            cfg.segmentation.xmem_init_with_sam = False
            if cfg.segmentation.backend.lower() != "xmem":
                raise ValueError("--manual-seed requires segmentation.backend == 'xmem'")
        voxel_size = cfg.pcd_voxel_size
        if args.use_segmentation:
            segmenter = build_segmenter(cfg.segmentation, device=cfg.device, num_cameras=len(camera_configs))
    elif args.use_segmentation:
        raise ValueError("--use-segmentation requires --use-config (to load checkpoints).")

    perception = RealSensePerception(camera_configs, segmenter=segmenter, voxel_size=voxel_size)
    if args.manual_seed:
        from ip.deployment.manual_seed_xmem import manual_seed_xmem
        serials = [cam.serial for cam in camera_configs]
        manual_seed_xmem(perception, serials, out_dir=args.manual_seed_out)

    server = viser.ViserServer()
    server.scene.world_axes.visible = bool(args.show_axes)

    if args.show_cameras:
        for cam in camera_configs:
            q = _quat_from_matrix(cam.T_world_camera[:3, :3])
            t = cam.T_world_camera[:3, 3]
            server.scene.add_frame(
                f"/camera/{cam.serial}",
                position=np.asarray(t, dtype=np.float32),
                wxyz=np.asarray(q, dtype=np.float32),
                axes_length=0.08,
                axes_radius=0.004,
            )

    if args.show_ee:
        if args.ee_from_demo is None:
            raise ValueError("--show-ee requires --ee-from-demo <frame_idx>")
        if not T_w_es:
            raise RuntimeError("Demo has no T_w_es; cannot show EE frame.")
        ee_idx = int(np.clip(args.ee_from_demo, 0, len(T_w_es) - 1))
        T_w_e = np.asarray(T_w_es[ee_idx], dtype=np.float64)
        q = _quat_from_matrix(T_w_e[:3, :3])
        t = T_w_e[:3, 3]
        server.scene.add_frame(
            "/ee",
            position=np.asarray(t, dtype=np.float32),
            wxyz=np.asarray(q, dtype=np.float32),
            axes_length=0.08,
            axes_radius=0.004,
        )

    demo_handle = server.scene.add_point_cloud(
        "/demo/pcd",
        points=demo_pts,
        colors=(180, 180, 180),
        point_size=0.003,
        point_shape="square",
    )

    live_handle = server.scene.add_point_cloud(
        "/live/pcd",
        points=np.zeros((0, 3), dtype=np.float32),
        colors=(80, 220, 120),
        point_size=0.003,
        point_shape="square",
    )

    tcp_handle = None
    if args.show_demo_tcp and T_w_es:
        tcp_pos = np.asarray(T_w_es[frame_idx], dtype=np.float32)[:3, 3]
        tcp_handle = server.scene.add_icosphere(
            "/demo/tcp",
            radius=args.tcp_radius,
            color=(255, 0, 0),
            position=tcp_pos,
        )

    def _capture_live() -> np.ndarray:
        live = perception.capture_pcd_world(use_segmentation=args.use_segmentation)
        live = _subsample(live, args.live_max_points)
        return live.astype(np.float32)

    def _update_demo(new_idx: int):
        nonlocal frame_idx
        frame_idx = int(np.clip(new_idx, 0, len(pcds) - 1))
        pts = np.asarray(pcds[frame_idx], dtype=np.float32)
        demo_handle.points = _subsample(pts, args.demo_max_points)
        if tcp_handle is not None and T_w_es:
            tcp_handle.position = np.asarray(T_w_es[frame_idx], dtype=np.float32)[:3, 3]

    with server.gui.add_folder("Alignment"):
        live_on = server.gui.add_checkbox("Live Update", initial_value=args.live_refresh_hz > 0)
        live_hz = server.gui.add_slider(
            "Live Hz",
            min=1,
            max=20,
            step=1,
            initial_value=int(args.live_refresh_hz) if args.live_refresh_hz > 0 else 2,
        )
        demo_slider = server.gui.add_slider("Demo Frame", min=0, max=len(pcds) - 1, step=1, initial_value=frame_idx)

    @demo_slider.on_update
    def _(_evt):
        _update_demo(demo_slider.value)

    def _live_loop():
        try:
            while True:
                if live_on.value:
                    live_handle.points = _capture_live()
                time.sleep(1.0 / max(1.0, float(live_hz.value)))
        finally:
            perception.stop()

    thread = threading.Thread(target=_live_loop, daemon=True)
    thread.start()
    server.sleep_forever()


if __name__ == "__main__":
    main()
