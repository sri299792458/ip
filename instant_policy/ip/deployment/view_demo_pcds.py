#!/usr/bin/env python3
import argparse
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


def _require_viser():
    if viser is None:
        raise ImportError(f"viser is required: {_VISER_IMPORT_ERROR}")


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx]


def _policy_subsample(points: np.ndarray, num_points: int) -> np.ndarray:
    from ip.utils.data_proc import subsample_pcd
    return subsample_pcd(points, num_points=num_points)


def _to_ee_frame(points: np.ndarray, T_w_e: np.ndarray) -> np.ndarray:
    R = T_w_e[:3, :3]
    t = T_w_e[:3, 3]
    return (R.T @ (points - t).T).T


def _load_config_defaults():
    import importlib.util
    entry = Path(__file__).resolve().parents[1] / "deployment.py"
    spec = importlib.util.spec_from_file_location("ip_deploy_entry", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_default_config()


def main():
    _require_viser()
    parser = argparse.ArgumentParser(description="Play back demo point clouds in Viser.")
    parser.add_argument("--demo", required=True, help="Path to demo .pkl")
    parser.add_argument("--stride", type=int, default=2, help="Show every Nth frame during playback")
    parser.add_argument("--fps", type=float, default=20.0, help="Playback speed (frames per second)")
    parser.add_argument("--max-points", type=int, default=120000, help="Max points per frame (non-policy view)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (inclusive, -1 for last)")
    parser.add_argument("--show-axes", action="store_true", help="Show world axes")
    parser.add_argument("--show-tcp", action="store_true", help="Show TCP position as a red sphere")
    parser.add_argument("--tcp-radius", type=float, default=0.01, help="TCP sphere radius (meters)")
    parser.add_argument("--policy-view", action="store_true", help="EE frame + subsample (policy input)")
    parser.add_argument("--waypoints", action="store_true", help="Use waypoint frames (training-style)")
    parser.add_argument("--num-points", type=int, default=None, help="Points per frame for policy view")
    parser.add_argument("--num-waypoints", type=int, default=None, help="Number of waypoints if --waypoints")
    parser.add_argument("--use-config", action="store_true", help="Load defaults from deployment.py")
    parser.add_argument("--point-size", type=float, default=0.003, help="Point size in meters")
    parser.add_argument(
        "--point-shape",
        default="square",
        choices=["square", "diamond", "circle", "rounded", "sparkle"],
        help="Point shape",
    )
    args = parser.parse_args()

    demo_path = Path(args.demo)
    with demo_path.open("rb") as f:
        data = pickle.load(f)

    pcds = data.get("pcds", [])
    T_w_es = data.get("T_w_es", [])
    grips = data.get("grips", [])
    valid_indices = [i for i, p in enumerate(pcds) if p is not None and len(p) > 0]
    if not valid_indices:
        raise RuntimeError("No valid point clouds found in demo.")

    if args.use_config:
        cfg = _load_config_defaults()
        if args.num_points is None:
            args.num_points = int(getattr(cfg, "pcd_num_points", 2048))
        if args.num_waypoints is None:
            args.num_waypoints = int(getattr(cfg, "num_traj_wp", 10))
    if args.num_points is None:
        args.num_points = 2048

    if args.waypoints:
        if not T_w_es or not grips:
            raise RuntimeError("Waypoints require T_w_es and grips in the demo.")
        from ip.utils.data_proc import extract_waypoints
        wp = extract_waypoints(np.array(T_w_es), np.array(grips), num_waypoints=args.num_waypoints or 10)
        valid_indices = [i for i in valid_indices if i in wp]

    start = max(0, args.start)
    end = args.end if args.end >= 0 else len(pcds) - 1
    end = min(end, len(pcds) - 1)
    stride = max(1, args.stride)
    valid_indices = [i for i in valid_indices if start <= i <= end]
    if not valid_indices:
        raise RuntimeError("No valid frames in the selected range.")

    def _frame_points(idx: int) -> np.ndarray:
        frame = pcds[idx]
        if frame is None or len(frame) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        pts = np.asarray(frame, dtype=np.float32)
        if args.policy_view and idx < len(T_w_es):
            pts = _to_ee_frame(pts, np.asarray(T_w_es[idx], dtype=np.float32))
            pts = _policy_subsample(pts, args.num_points).astype(np.float32)
        else:
            pts = _subsample(pts, args.max_points).astype(np.float32)
        return pts

    server = viser.ViserServer()
    server.scene.world_axes.visible = bool(args.show_axes)

    first_idx = valid_indices[0]
    pts0 = _frame_points(first_idx)
    colors = (180, 180, 180)

    pcd_handle = server.scene.add_point_cloud(
        "/demo/pcd",
        points=pts0,
        colors=colors,
        point_size=args.point_size,
        point_shape=args.point_shape,
        precision="float32",
    )

    tcp_handle = None
    if args.show_tcp and T_w_es:
        pos = np.zeros(3) if args.policy_view else np.asarray(T_w_es[first_idx])[:3, 3]
        tcp_handle = server.scene.add_icosphere(
            "/demo/tcp",
            radius=args.tcp_radius,
            color=(255, 0, 0),
            position=pos,
        )

    with server.gui.add_folder("Playback"):
        play = server.gui.add_checkbox("Play", initial_value=False)
        fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=int(args.fps))
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(valid_indices) - 1,
            step=1,
            initial_value=0,
        )
        frame_idx_readout = server.gui.add_number("Frame idx", initial_value=first_idx, disabled=True)

    state = {"frame": 0, "lock": threading.Lock(), "updating": False}

    def _set_frame(frame_list_idx: int, from_slider: bool = False) -> None:
        with state["lock"]:
            frame_list_idx = int(np.clip(frame_list_idx, 0, len(valid_indices) - 1))
            state["frame"] = frame_list_idx
            frame_idx = valid_indices[frame_list_idx]
            pts = _frame_points(frame_idx)
            pcd_handle.points = pts
            if tcp_handle is not None and frame_idx < len(T_w_es):
                pos = np.zeros(3) if args.policy_view else np.asarray(T_w_es[frame_idx])[:3, 3]
                tcp_handle.position = pos
            frame_idx_readout.value = frame_idx
            if not from_slider:
                state["updating"] = True
                frame_slider.value = frame_list_idx
                state["updating"] = False

    @frame_slider.on_update
    def _(_evt):
        if state["updating"]:
            return
        _set_frame(frame_slider.value, from_slider=True)

    def _playback_loop():
        while True:
            if play.value:
                _set_frame(state["frame"] + stride)
            time.sleep(1.0 / max(1.0, float(fps.value)))

    thread = threading.Thread(target=_playback_loop, daemon=True)
    thread.start()

    server.sleep_forever()


if __name__ == "__main__":
    main()
