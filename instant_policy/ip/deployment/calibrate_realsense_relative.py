#!/usr/bin/env python3
"""
Estimate T_world_camera for a new RealSense using a reference camera + ArUco/Charuco target.

Assumes:
  - Reference camera already has T_world_camera in a calibration JSON.
  - Both cameras see the same static target during sampling.

Outputs a calibration JSON that includes the new camera's T_world_camera.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    if not hasattr(cv2, "aruco"):
        raise ImportError("cv2.aruco is missing. Install opencv-contrib-python.")


def _aruco_dict(name: str):
    mapping = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    if name not in mapping:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(mapping[name])


def _build_charuco_board(
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_size: float,
    dictionary,
):
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_size, marker_size, dictionary
        )
    if hasattr(cv2.aruco, "CharucoBoard"):
        # Some OpenCV builds require size as a tuple, others accept separate ints.
        try:
            return cv2.aruco.CharucoBoard(
                (squares_x, squares_y), square_size, marker_size, dictionary
            )
        except (TypeError, cv2.error):
            return cv2.aruco.CharucoBoard(
                squares_x, squares_y, square_size, marker_size, dictionary
            )
    raise RuntimeError("Charuco board is not supported by this OpenCV build.")


def _marker_object_points(tag_size: float) -> np.ndarray:
    s = tag_size
    # Order must match ArUco corner order: tl, tr, br, bl
    return np.array(
        [
            [-s / 2, s / 2, 0.0],
            [s / 2, s / 2, 0.0],
            [s / 2, -s / 2, 0.0],
            [-s / 2, -s / 2, 0.0],
        ],
        dtype=np.float32,
    )


def _detect_marker_pose(
    bgr: np.ndarray,
    dictionary,
    parameters,
    K: np.ndarray,
    dist: np.ndarray,
    tag_id: int,
    tag_size: float,
    return_debug: bool = False,
):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    info = {"num_markers": 0, "ids": [], "reproj_err": None, "used_id": False}
    if ids is None:
        return (None, info, corners, ids) if return_debug else None
    ids_raw = ids
    ids = ids.flatten()
    info["num_markers"] = int(len(ids))
    info["ids"] = [int(x) for x in ids.tolist()]
    if tag_id not in ids:
        return (None, info, corners, ids) if return_debug else None
    idx = int(np.where(ids == tag_id)[0][0])
    info["used_id"] = True
    marker_corners = corners[idx]

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        [marker_corners], tag_size, K, dist
    )
    rvec = rvecs[0, 0]
    tvec = tvecs[0, 0]

    obj_pts = _marker_object_points(tag_size)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    obs = marker_corners.reshape(-1, 2)
    reproj_err = float(np.linalg.norm(proj - obs, axis=1).mean())
    info["reproj_err"] = reproj_err
    if return_debug:
        return (rvec, tvec, reproj_err), info, corners, ids_raw
    return rvec, tvec, reproj_err


def _detect_charuco_pose(
    bgr: np.ndarray,
    dictionary,
    parameters,
    board,
    K: np.ndarray,
    dist: np.ndarray,
    return_debug: bool = False,
):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    info = {"num_markers": 0, "num_charuco": 0, "ids": [], "reproj_err": None}
    if ids is None or len(ids) == 0:
        return (None, info, corners, ids, None, None) if return_debug else None
    ids_raw = ids
    ids = ids.flatten()
    info["num_markers"] = int(len(ids))
    info["ids"] = [int(x) for x in ids.tolist()]

    if not hasattr(cv2.aruco, "interpolateCornersCharuco"):
        raise RuntimeError("Charuco detection requires cv2.aruco.interpolateCornersCharuco.")

    used_intrinsics = False
    try:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids_raw, gray, board, K, dist
        )
        used_intrinsics = True
    except Exception:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids_raw, gray, board
        )
    if charuco_ids is None or len(charuco_ids) < 4:
        if charuco_ids is not None:
            info["num_charuco"] = int(len(charuco_ids))
        return (None, info, corners, ids_raw, charuco_corners, charuco_ids) if return_debug else None
    info["num_charuco"] = int(len(charuco_ids))
    info["used_intrinsics"] = used_intrinsics

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, dist, None, None
    )
    if not ok:
        return (None, info, corners, ids_raw, charuco_corners, charuco_ids) if return_debug else None

    rvec = np.array(rvec, dtype=np.float64).reshape(3)
    tvec = np.array(tvec, dtype=np.float64).reshape(3)

    if hasattr(board, "chessboardCorners"):
        obj_pts = board.chessboardCorners[charuco_ids.flatten()]
    else:
        obj_pts = board.getChessboardCorners()[charuco_ids.flatten()]
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    obs = charuco_corners.reshape(-1, 2)
    reproj_err = float(np.linalg.norm(proj - obs, axis=1).mean())
    info["reproj_err"] = reproj_err
    if return_debug:
        return (rvec, tvec, reproj_err), info, corners, ids_raw, charuco_corners, charuco_ids
    return rvec, tvec, reproj_err


def _charuco_area_ratio(charuco_corners, image_shape) -> float:
    if charuco_corners is None:
        return 0.0
    pts = charuco_corners.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 3:
        return 0.0
    hull = cv2.convexHull(pts)
    area = float(cv2.contourArea(hull))
    h, w = image_shape[:2]
    img_area = float(h * w)
    if img_area <= 0:
        return 0.0
    return area / img_area


def _board_view_angle_deg(rvec: np.ndarray) -> Tuple[float, bool]:
    R, _ = cv2.Rodrigues(rvec)
    n_cam = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_norm = float(np.linalg.norm(n_cam))
    if n_norm < 1e-12:
        return 180.0, False
    n_cam = n_cam / n_norm
    cos = float(np.clip(n_cam[2], -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos)))
    facing = n_cam[2] > 0
    return angle, facing


def _get_synced_color_frames(
    pipe_ref,
    pipe_new,
    max_dt_ms: float,
    max_tries: int,
):
    frames_ref = pipe_ref.wait_for_frames()
    frames_new = pipe_new.wait_for_frames()
    color_ref = frames_ref.get_color_frame()
    color_new = frames_new.get_color_frame()
    if not color_ref or not color_new:
        return None, None, None, None
    domain_ref = color_ref.get_frame_timestamp_domain()
    domain_new = color_new.get_frame_timestamp_domain()
    system_domain = getattr(rs.timestamp_domain, "system_time", None)
    if system_domain is None or domain_ref != domain_new or domain_ref != system_domain:
        return color_ref, color_new, None, (domain_ref, domain_new)
    ts_ref = float(color_ref.get_timestamp())
    ts_new = float(color_new.get_timestamp())
    dt = abs(ts_ref - ts_new)

    tries = 0
    while dt > max_dt_ms and tries < max_tries:
        if ts_ref < ts_new:
            frames_ref = pipe_ref.wait_for_frames()
            color_ref = frames_ref.get_color_frame()
            if not color_ref:
                return None, None, None, None
            ts_ref = float(color_ref.get_timestamp())
        else:
            frames_new = pipe_new.wait_for_frames()
            color_new = frames_new.get_color_frame()
            if not color_new:
                return None, None, None, None
            ts_new = float(color_new.get_timestamp())
        dt = abs(ts_ref - ts_new)
        tries += 1

    return color_ref, color_new, dt, None


def _draw_debug_image(
    bgr: np.ndarray,
    corners,
    ids,
    charuco_corners=None,
    charuco_ids=None,
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    rvec: Optional[np.ndarray] = None,
    tvec: Optional[np.ndarray] = None,
) -> np.ndarray:
    vis = bgr.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
    if rvec is not None and tvec is not None and K is not None and dist is not None:
        if hasattr(cv2, "drawFrameAxes"):
            rvec_cv = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec_cv = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            cv2.drawFrameAxes(vis, K, dist, rvec_cv, tvec_cv, 0.05)
    return vis

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


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx + yy - zz],
        ],
        dtype=np.float64,
    )


def _average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    A = np.zeros((4, 4), dtype=np.float64)
    q_ref = None
    for R in rotations:
        R = _project_to_so3(R)
        q = _quat_from_matrix(R)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            continue
        q = q / q_norm
        if q_ref is None:
            q_ref = q
        elif np.dot(q, q_ref) < 0:
            q = -q
        A += np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(A)
    q_mean = eigvecs[:, np.argmax(eigvals)]
    q_mean = q_mean / np.linalg.norm(q_mean)
    return _project_to_so3(_matrix_from_quat(q_mean))


def _transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj


def _start_pipeline(serial: str, width: int, height: int, fps: int, enable_global_time: bool):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    if enable_global_time:
        try:
            dev = profile.get_device()
            for sensor in dev.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
        except Exception:
            pass
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream.get_intrinsics()
    K = np.array(
        [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.array(intr.coeffs, dtype=np.float64)
    return pipeline, K, dist


def _load_T_world_camera(calib_path: Path, serial: str) -> np.ndarray:
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with calib_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cams = data.get("cameras", {})
    if serial not in cams:
        raise KeyError(f"Serial {serial} not found in {calib_path}")
    T = np.array(cams[serial].get("T_world_camera"), dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_world_camera for {serial} is not 4x4")
    return T


def _update_calib_json(
    calib_path: Path,
    out_path: Path,
    tag_meta: Dict,
    ref_serial: str,
    T_world_ref: np.ndarray,
    new_serial: str,
    stats: Dict,
) -> None:
    if calib_path.exists():
        with calib_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    # Replace tag metadata to avoid stale fields (e.g., prior ArUco id/size).
    data["tag"] = dict(tag_meta)
    data.setdefault("cameras", {})

    if ref_serial not in data["cameras"]:
        data["cameras"][ref_serial] = {"T_world_camera": T_world_ref.tolist()}

    data["cameras"][new_serial] = {
        "T_world_camera": stats["T_world_camera"].tolist(),
        "num_samples": stats["num_samples"],
        "used_frames": stats["used_frames"],
        "reproj_error_ref_mean_px": stats["reproj_error_ref_mean_px"],
        "reproj_error_ref_std_px": stats["reproj_error_ref_std_px"],
        "reproj_error_new_mean_px": stats["reproj_error_new_mean_px"],
        "reproj_error_new_std_px": stats["reproj_error_new_std_px"],
        "rot_error_mean_deg": stats["rot_error_mean_deg"],
        "rot_error_std_deg": stats["rot_error_std_deg"],
        "translation_std_m": stats["translation_std_m"],
    }
    if "sync_delta_ms_mean" in stats:
        data["cameras"][new_serial]["sync_delta_ms_mean"] = stats["sync_delta_ms_mean"]
    if "sync_delta_ms_std" in stats:
        data["cameras"][new_serial]["sync_delta_ms_std"] = stats["sync_delta_ms_std"]
    if "T_ref_new" in stats:
        data["cameras"][new_serial]["T_ref_new"] = stats["T_ref_new"].tolist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    _require_deps()
    parser = argparse.ArgumentParser(
        description="Calibrate a new RealSense using a reference camera + ArUco/Charuco target."
    )
    parser.add_argument("--ref-serial", required=True, help="Reference camera serial")
    parser.add_argument("--new-serial", required=True, help="New camera serial")
    parser.add_argument(
        "--calib",
        default=str(Path(__file__).resolve().parent / "calibration_outputs" / "realsense_T_world_camera.json"),
        help="Path to existing calibration JSON (contains ref T_world_camera)",
    )
    parser.add_argument("--tag-dict", default="DICT_4X4_50", help="ArUco dictionary name")
    parser.add_argument("--tag-id", type=int, default=1, help="ArUco marker ID")
    parser.add_argument(
        "--tag-size",
        type=float,
        default=None,
        help="ArUco marker edge length (meters). Required unless --charuco is set.",
    )
    parser.add_argument(
        "--charuco",
        action="store_true",
        help="Use a Charuco board instead of a single ArUco marker.",
    )
    parser.add_argument(
        "--charuco-squares-x",
        type=int,
        default=None,
        help="Number of squares in X (columns) for Charuco board.",
    )
    parser.add_argument(
        "--charuco-squares-y",
        type=int,
        default=None,
        help="Number of squares in Y (rows) for Charuco board.",
    )
    parser.add_argument(
        "--charuco-square-size",
        type=float,
        default=None,
        help="Charuco square size (meters).",
    )
    parser.add_argument(
        "--charuco-marker-size",
        type=float,
        default=None,
        help="Charuco marker size (meters).",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument(
        "--enable-global-time",
        action="store_true",
        help="Enable RealSense global time so timestamps are comparable across devices.",
    )
    parser.add_argument(
        "--sync-frames",
        action="store_true",
        help="Synchronize ref/new frames by timestamp before processing.",
    )
    parser.add_argument(
        "--no-sync-frames",
        dest="sync_frames",
        action="store_false",
        help="Disable timestamp pairing (not recommended).",
    )
    parser.set_defaults(sync_frames=True)
    parser.add_argument(
        "--max-sync-delta-ms",
        type=float,
        default=20.0,
        help="Max allowed timestamp delta between ref/new frames (ms).",
    )
    parser.add_argument(
        "--sync-tries",
        type=int,
        default=30,
        help="Max extra frame pulls to reach sync tolerance.",
    )
    parser.add_argument(
        "--min-charuco-corners",
        type=int,
        default=12,
        help="Minimum Charuco corners required per camera.",
    )
    parser.add_argument(
        "--min-board-area",
        type=float,
        default=0.01,
        help="Minimum Charuco convex hull area as fraction of image area.",
    )
    parser.add_argument(
        "--max-view-angle-deg",
        type=float,
        default=60.0,
        help="Max angle between board normal and camera optical axis.",
    )
    parser.add_argument(
        "--min-pose-rot-deg",
        type=float,
        default=0.0,
        help="Require minimum rotation between accepted board poses (degrees). Set 0 to disable.",
    )
    parser.add_argument(
        "--min-pose-trans-m",
        type=float,
        default=0.0,
        help="Require minimum translation between accepted board poses (meters). Set 0 to disable.",
    )
    parser.add_argument(
        "--require-board-front",
        action="store_true",
        help="Require board normal to face camera (n_cam.z > 0).",
    )
    parser.add_argument(
        "--allow-board-back",
        dest="require_board_front",
        action="store_false",
        help="Allow back-facing board poses.",
    )
    parser.set_defaults(require_board_front=True)
    parser.add_argument("--debug", action="store_true", help="Print per-frame detection stats.")
    parser.add_argument(
        "--debug-every",
        type=int,
        default=10,
        help="Print debug info every N frames (default: 10).",
    )
    parser.add_argument(
        "--debug-save-dir",
        type=str,
        default=None,
        help="If set, save debug images to this directory (PNG).",
    )
    parser.add_argument(
        "--debug-save-mode",
        type=str,
        choices=("fail", "all"),
        default="fail",
        help="Save debug images for failed frames only or all frames.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: calibration_outputs/realsense_T_world_camera_<new>.json)",
    )
    args = parser.parse_args()

    calib_path = Path(args.calib)
    T_world_ref = _load_T_world_camera(calib_path, args.ref_serial)

    dictionary = _aruco_dict(args.tag_dict)
    if args.charuco:
        if (
            args.charuco_squares_x is None
            or args.charuco_squares_y is None
            or args.charuco_square_size is None
            or args.charuco_marker_size is None
        ):
            raise ValueError(
                "--charuco requires --charuco-squares-x, --charuco-squares-y, "
                "--charuco-square-size, and --charuco-marker-size."
            )
        board = _build_charuco_board(
            args.charuco_squares_x,
            args.charuco_squares_y,
            args.charuco_square_size,
            args.charuco_marker_size,
            dictionary,
        )
    else:
        if args.tag_size is None:
            raise ValueError("--tag-size is required when not using --charuco.")
        board = None
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    pipe_ref, K_ref, dist_ref = _start_pipeline(
        args.ref_serial, args.width, args.height, args.fps, args.enable_global_time
    )
    pipe_new, K_new, dist_new = _start_pipeline(
        args.new_serial, args.width, args.height, args.fps, args.enable_global_time
    )
    debug_dir = Path(args.debug_save_dir) if args.debug_save_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        for _ in range(args.warmup_frames):
            pipe_ref.wait_for_frames()
            pipe_new.wait_for_frames()

        rotations = []
        translations = []
        reproj_ref = []
        reproj_new = []
        sync_deltas = []
        ref_new_rotations = []
        ref_new_translations = []
        used_frames = 0
        T_world_tags = []
        last_ref_pose = None

        warned_sync_domain = False
        while len(rotations) < args.num_samples and used_frames < args.max_frames:
            if args.sync_frames:
                color_ref, color_new, dt_ms, domain_info = _get_synced_color_frames(
                    pipe_ref, pipe_new, args.max_sync_delta_ms, args.sync_tries
                )
            else:
                frames_ref = pipe_ref.wait_for_frames()
                frames_new = pipe_new.wait_for_frames()
                color_ref = frames_ref.get_color_frame()
                color_new = frames_new.get_color_frame()
                dt_ms = None
                domain_info = None
            used_frames += 1

            if not color_ref or not color_new:
                continue
            if args.sync_frames and dt_ms is None and domain_info is not None and not warned_sync_domain:
                warned_sync_domain = True
                print(
                    "Warning: timestamp domains are not comparable; sync disabled for this run. "
                    "Use --enable-global-time or --no-sync-frames."
                )
            if dt_ms is not None:
                sync_deltas.append(float(dt_ms))
                if dt_ms > args.max_sync_delta_ms:
                    if args.debug and used_frames % max(args.debug_every, 1) == 0:
                        print(f"[frame {used_frames}] sync_fail dt_ms={dt_ms:.1f}")
                    continue

            bgr_ref = np.asanyarray(color_ref.get_data())
            bgr_new = np.asanyarray(color_new.get_data())

            if args.charuco:
                pose_ref, info_ref, corners_ref, ids_ref, cc_ref, ci_ref = _detect_charuco_pose(
                    bgr_ref, dictionary, parameters, board, K_ref, dist_ref, return_debug=True
                )
                pose_new, info_new, corners_new, ids_new, cc_new, ci_new = _detect_charuco_pose(
                    bgr_new, dictionary, parameters, board, K_new, dist_new, return_debug=True
                )
            else:
                if args.debug or debug_dir is not None:
                    pose_ref, info_ref, corners_ref, ids_ref = _detect_marker_pose(
                        bgr_ref,
                        dictionary,
                        parameters,
                        K_ref,
                        dist_ref,
                        args.tag_id,
                        args.tag_size,
                        return_debug=True,
                    )
                    pose_new, info_new, corners_new, ids_new = _detect_marker_pose(
                        bgr_new,
                        dictionary,
                        parameters,
                        K_new,
                        dist_new,
                        args.tag_id,
                        args.tag_size,
                        return_debug=True,
                    )
                    cc_ref = ci_ref = cc_new = ci_new = None
                else:
                    pose_ref = _detect_marker_pose(
                        bgr_ref,
                        dictionary,
                        parameters,
                        K_ref,
                        dist_ref,
                        args.tag_id,
                        args.tag_size,
                    )
                    pose_new = _detect_marker_pose(
                        bgr_new,
                        dictionary,
                        parameters,
                        K_new,
                        dist_new,
                        args.tag_id,
                        args.tag_size,
                    )
            if pose_ref is None or pose_new is None:
                if args.debug and used_frames % max(args.debug_every, 1) == 0:
                    ref_dbg = info_ref if args.debug or debug_dir is not None else {}
                    new_dbg = info_new if args.debug or debug_dir is not None else {}
                    print(
                        f"[frame {used_frames}] pose_missing "
                        f"ref markers={ref_dbg.get('num_markers', 0)} "
                        f"charuco={ref_dbg.get('num_charuco', 0)} "
                        f"new markers={new_dbg.get('num_markers', 0)} "
                        f"charuco={new_dbg.get('num_charuco', 0)}"
                    )
                if debug_dir is not None and args.debug_save_mode in ("fail", "all"):
                    ref_vis = _draw_debug_image(
                        bgr_ref,
                        corners_ref if args.debug or debug_dir is not None else None,
                        ids_ref if args.debug or debug_dir is not None else None,
                        charuco_corners=cc_ref if args.debug or debug_dir is not None else None,
                        charuco_ids=ci_ref if args.debug or debug_dir is not None else None,
                        K=K_ref,
                        dist=dist_ref,
                    )
                    new_vis = _draw_debug_image(
                        bgr_new,
                        corners_new if args.debug or debug_dir is not None else None,
                        ids_new if args.debug or debug_dir is not None else None,
                        charuco_corners=cc_new if args.debug or debug_dir is not None else None,
                        charuco_ids=ci_new if args.debug or debug_dir is not None else None,
                        K=K_new,
                        dist=dist_new,
                    )
                    cv2.imwrite(str(debug_dir / f"frame_{used_frames:06d}_ref.png"), ref_vis)
                    cv2.imwrite(str(debug_dir / f"frame_{used_frames:06d}_new.png"), new_vis)
                continue
            rvec_ref, tvec_ref, err_ref = pose_ref
            rvec_new, tvec_new, err_new = pose_new
            if err_ref > args.max_reproj_error or err_new > args.max_reproj_error:
                if args.debug and used_frames % max(args.debug_every, 1) == 0:
                    print(
                        f"[frame {used_frames}] reproj_fail "
                        f"ref={err_ref:.2f}px new={err_new:.2f}px "
                        f"(max {args.max_reproj_error:.2f}px)"
                    )
                if debug_dir is not None and args.debug_save_mode in ("fail", "all"):
                    ref_vis = _draw_debug_image(
                        bgr_ref,
                        corners_ref if args.debug or debug_dir is not None else None,
                        ids_ref if args.debug or debug_dir is not None else None,
                        charuco_corners=cc_ref if args.debug or debug_dir is not None else None,
                        charuco_ids=ci_ref if args.debug or debug_dir is not None else None,
                        K=K_ref,
                        dist=dist_ref,
                        rvec=rvec_ref,
                        tvec=tvec_ref,
                    )
                    new_vis = _draw_debug_image(
                        bgr_new,
                        corners_new if args.debug or debug_dir is not None else None,
                        ids_new if args.debug or debug_dir is not None else None,
                        charuco_corners=cc_new if args.debug or debug_dir is not None else None,
                        charuco_ids=ci_new if args.debug or debug_dir is not None else None,
                        K=K_new,
                        dist=dist_new,
                        rvec=rvec_new,
                        tvec=tvec_new,
                    )
                    cv2.imwrite(str(debug_dir / f"frame_{used_frames:06d}_ref.png"), ref_vis)
                    cv2.imwrite(str(debug_dir / f"frame_{used_frames:06d}_new.png"), new_vis)
                continue
            if args.charuco:
                if info_ref.get("num_charuco", 0) < args.min_charuco_corners or info_new.get(
                    "num_charuco", 0
                ) < args.min_charuco_corners:
                    if args.debug and used_frames % max(args.debug_every, 1) == 0:
                        print(
                            f"[frame {used_frames}] corners_fail "
                            f"ref={info_ref.get('num_charuco', 0)} "
                            f"new={info_new.get('num_charuco', 0)} "
                            f"(min {args.min_charuco_corners})"
                        )
                    continue
                area_ref = _charuco_area_ratio(cc_ref, bgr_ref.shape)
                area_new = _charuco_area_ratio(cc_new, bgr_new.shape)
                if area_ref < args.min_board_area or area_new < args.min_board_area:
                    if args.debug and used_frames % max(args.debug_every, 1) == 0:
                        print(
                            f"[frame {used_frames}] area_fail "
                            f"ref={area_ref:.4f} new={area_new:.4f} "
                            f"(min {args.min_board_area:.4f})"
                        )
                    continue
                angle_ref, facing_ref = _board_view_angle_deg(rvec_ref)
                angle_new, facing_new = _board_view_angle_deg(rvec_new)
                if angle_ref > args.max_view_angle_deg or angle_new > args.max_view_angle_deg:
                    if args.debug and used_frames % max(args.debug_every, 1) == 0:
                        print(
                            f"[frame {used_frames}] angle_fail "
                            f"ref={angle_ref:.1f} new={angle_new:.1f} "
                            f"(max {args.max_view_angle_deg:.1f})"
                        )
                    continue
                if args.require_board_front and (not facing_ref or not facing_new):
                    if args.debug and used_frames % max(args.debug_every, 1) == 0:
                        print(
                            f"[frame {used_frames}] facing_fail "
                            f"ref_front={facing_ref} new_front={facing_new}"
                        )
                    continue
            T_cam_ref_tag = _transform_from_rvec_tvec(rvec_ref, tvec_ref)
            T_cam_new_tag = _transform_from_rvec_tvec(rvec_new, tvec_new)
            if args.min_pose_rot_deg > 0.0 or args.min_pose_trans_m > 0.0:
                if last_ref_pose is not None:
                    T_delta = np.linalg.inv(last_ref_pose) @ T_cam_ref_tag
                    rot_delta = _rotation_angle_deg(T_delta[:3, :3])
                    trans_delta = float(np.linalg.norm(T_delta[:3, 3]))
                    if rot_delta < args.min_pose_rot_deg and trans_delta < args.min_pose_trans_m:
                        if args.debug and used_frames % max(args.debug_every, 1) == 0:
                            print(
                                f"[frame {used_frames}] pose_static "
                                f"rot={rot_delta:.2f}deg trans={trans_delta:.3f}m "
                                f"(min {args.min_pose_rot_deg:.2f}deg, {args.min_pose_trans_m:.3f}m)"
                            )
                        continue
                last_ref_pose = T_cam_ref_tag
            if args.debug and used_frames % max(args.debug_every, 1) == 0:
                ref_dbg = info_ref if args.debug or debug_dir is not None else {}
                new_dbg = info_new if args.debug or debug_dir is not None else {}
                angle_ref = angle_new = None
                area_ref = area_new = None
                if args.charuco:
                    try:
                        angle_ref, _ = _board_view_angle_deg(rvec_ref)
                        angle_new, _ = _board_view_angle_deg(rvec_new)
                        area_ref = _charuco_area_ratio(cc_ref, bgr_ref.shape)
                        area_new = _charuco_area_ratio(cc_new, bgr_new.shape)
                    except Exception:
                        angle_ref = angle_new = None
                        area_ref = area_new = None
                dt_msg = f" dt_ms={dt_ms:.1f}" if dt_ms is not None else ""
                angle_msg = ""
                if angle_ref is not None and angle_new is not None:
                    angle_msg = f" angle_ref={angle_ref:.1f} angle_new={angle_new:.1f}"
                area_msg = ""
                if area_ref is not None and area_new is not None:
                    area_msg = f" area_ref={area_ref:.3f} area_new={area_new:.3f}"
                print(
                    f"[frame {used_frames}] ok"
                    f"{dt_msg}"
                    f"{angle_msg}"
                    f"{area_msg} "
                    f"ref markers={ref_dbg.get('num_markers', 0)} "
                    f"charuco={ref_dbg.get('num_charuco', 0)} "
                    f"err={err_ref:.2f}px "
                    f"new markers={new_dbg.get('num_markers', 0)} "
                    f"charuco={new_dbg.get('num_charuco', 0)} "
                    f"err={err_new:.2f}px"
                )
            if debug_dir is not None and args.debug_save_mode == "all":
                ref_vis = _draw_debug_image(
                    bgr_ref,
                    corners_ref if args.debug or debug_dir is not None else None,
                    ids_ref if args.debug or debug_dir is not None else None,
                    charuco_corners=cc_ref if args.debug or debug_dir is not None else None,
                    charuco_ids=ci_ref if args.debug or debug_dir is not None else None,
                    K=K_ref,
                    dist=dist_ref,
                    rvec=rvec_ref,
                    tvec=tvec_ref,
                )
                new_vis = _draw_debug_image(
                    bgr_new,
                    corners_new if args.debug or debug_dir is not None else None,
                    ids_new if args.debug or debug_dir is not None else None,
                    charuco_corners=cc_new if args.debug or debug_dir is not None else None,
                    charuco_ids=ci_new if args.debug or debug_dir is not None else None,
                    K=K_new,
                    dist=dist_new,
                    rvec=rvec_new,
                    tvec=tvec_new,
                )
                cv2.imwrite(str(debug_dir / f"frame_{used_frames:06d}_ref.png"), ref_vis)
                cv2.imwrite(str(debug_dir / f"frame_{used_frames:06d}_new.png"), new_vis)

            # world<-tag from reference camera
            T_world_tag = T_world_ref @ T_cam_ref_tag
            T_world_tags.append(T_world_tag)

            # world<-new camera
            T_world_new = T_world_tag @ np.linalg.inv(T_cam_new_tag)

            rotations.append(T_world_new[:3, :3])
            translations.append(T_world_new[:3, 3])
            reproj_ref.append(err_ref)
            reproj_new.append(err_new)

            # ref<-new camera directly (camera-to-camera)
            T_ref_new = T_cam_ref_tag @ np.linalg.inv(T_cam_new_tag)
            ref_new_rotations.append(T_ref_new[:3, :3])
            ref_new_translations.append(T_ref_new[:3, 3])

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

        if not rotations:
            raise RuntimeError(
                "No valid detections. Check tag visibility, size, dictionary, or reprojection threshold."
            )

        R_mean = _average_rotations(rotations)
        t_mean = np.mean(np.stack(translations, axis=0), axis=0)
        T_mean = np.eye(4, dtype=np.float64)
        T_mean[:3, :3] = R_mean
        T_mean[:3, 3] = t_mean

        world_rot_errors = [_rotation_angle_deg(R_mean.T @ R_i) for R_i in rotations]
        rot_errors = world_rot_errors

        T_ref_new_mean = None
        ref_new_rot_errors = None
        ref_new_trans_std = None
        if ref_new_rotations:
            R_ref_new_mean = _average_rotations(ref_new_rotations)
            t_ref_new_mean = np.mean(np.stack(ref_new_translations, axis=0), axis=0)
            T_ref_new_mean = np.eye(4, dtype=np.float64)
            T_ref_new_mean[:3, :3] = R_ref_new_mean
            T_ref_new_mean[:3, 3] = t_ref_new_mean
            ref_new_rot_errors = [
                _rotation_angle_deg(R_ref_new_mean.T @ R_i) for R_i in ref_new_rotations
            ]
            ref_new_trans_std = np.std(np.stack(ref_new_translations, axis=0), axis=0).tolist()
            # Prefer ref->new mean for world output (more stable).
            T_mean = T_world_ref @ T_ref_new_mean
            if ref_new_rot_errors:
                rot_errors = ref_new_rot_errors

        T_world_tag_mean = None
        if T_world_tags:
            R_tag_mean = _average_rotations([T[:3, :3] for T in T_world_tags])
            t_tag_mean = np.mean(np.stack([T[:3, 3] for T in T_world_tags], axis=0), axis=0)
            T_world_tag_mean = np.eye(4, dtype=np.float64)
            T_world_tag_mean[:3, :3] = R_tag_mean
            T_world_tag_mean[:3, 3] = t_tag_mean

        stats = {
            "T_world_camera": T_mean,
            "num_samples": len(rotations),
            "used_frames": used_frames,
            "reproj_error_ref_mean_px": float(np.mean(reproj_ref)),
            "reproj_error_ref_std_px": float(np.std(reproj_ref)),
            "reproj_error_new_mean_px": float(np.mean(reproj_new)),
            "reproj_error_new_std_px": float(np.std(reproj_new)),
            "rot_error_mean_deg": float(np.mean(rot_errors)),
            "rot_error_std_deg": float(np.std(rot_errors)),
            "translation_std_m": (
                ref_new_trans_std
                if ref_new_trans_std is not None
                else np.std(np.stack(translations, axis=0), axis=0).tolist()
            ),
        }
        if sync_deltas:
            stats["sync_delta_ms_mean"] = float(np.mean(sync_deltas))
            stats["sync_delta_ms_std"] = float(np.std(sync_deltas))
        if T_ref_new_mean is not None:
            stats["T_ref_new"] = T_ref_new_mean
            if ref_new_rot_errors is not None:
                stats["ref_new_rot_error_mean_deg"] = float(np.mean(ref_new_rot_errors))
                stats["ref_new_rot_error_std_deg"] = float(np.std(ref_new_rot_errors))
            if ref_new_trans_std is not None:
                stats["ref_new_translation_std_m"] = ref_new_trans_std
            if world_rot_errors:
                stats["world_rot_error_mean_deg"] = float(np.mean(world_rot_errors))
                stats["world_rot_error_std_deg"] = float(np.std(world_rot_errors))

        tag_meta = {"dict": args.tag_dict}
        if args.charuco:
            tag_meta.update(
                {
                    "type": "charuco",
                    "squares_x": args.charuco_squares_x,
                    "squares_y": args.charuco_squares_y,
                    "square_size_m": args.charuco_square_size,
                    "marker_size_m": args.charuco_marker_size,
                }
            )
        else:
            tag_meta.update({"type": "aruco", "id": args.tag_id, "size_m": args.tag_size})
        if T_world_tag_mean is not None:
            tag_meta["T_world_tag"] = T_world_tag_mean.tolist()

        if args.out:
            out_path = Path(args.out)
        else:
            out_dir = Path(__file__).resolve().parent / "calibration_outputs"
            out_path = out_dir / f"realsense_T_world_camera_{args.new_serial}.json"

        _update_calib_json(
            calib_path=calib_path,
            out_path=out_path,
            tag_meta=tag_meta,
            ref_serial=args.ref_serial,
            T_world_ref=T_world_ref,
            new_serial=args.new_serial,
            stats=stats,
        )

        print(f"\nReference serial {args.ref_serial} T_world_camera:")
        print(T_world_ref)
        print(f"\nNew serial {args.new_serial} T_world_camera:")
        print(T_mean)
        print(
            "  samples:",
            stats["num_samples"],
            "reproj_ref(px):",
            f"{stats['reproj_error_ref_mean_px']:.2f}±{stats['reproj_error_ref_std_px']:.2f}",
            "reproj_new(px):",
            f"{stats['reproj_error_new_mean_px']:.2f}±{stats['reproj_error_new_std_px']:.2f}",
            "rot_err(deg):",
            f"{stats['rot_error_mean_deg']:.2f}±{stats['rot_error_std_deg']:.2f}",
        )
        if "sync_delta_ms_mean" in stats:
            print(
                "  sync_dt(ms):",
                f"{stats['sync_delta_ms_mean']:.1f}±{stats['sync_delta_ms_std']:.1f}",
            )
        if "T_ref_new" in stats:
            print(f"\nRef->New camera T_ref_new (ref frame):")
            print(stats["T_ref_new"])
            if "ref_new_rot_error_mean_deg" in stats:
                print(
                    "  ref_new_rot_err(deg):",
                    f"{stats['ref_new_rot_error_mean_deg']:.2f}±{stats['ref_new_rot_error_std_deg']:.2f}",
                )
            if "ref_new_translation_std_m" in stats:
                print("  ref_new_translation_std_m:", stats["ref_new_translation_std_m"])
            if "world_rot_error_mean_deg" in stats:
                print(
                    "  world_rot_err(deg):",
                    f"{stats['world_rot_error_mean_deg']:.2f}±{stats['world_rot_error_std_deg']:.2f}",
                )
        print(f"\nSaved calibration to {out_path}")
    finally:
        try:
            pipe_ref.stop()
        except Exception:
            pass
        try:
            pipe_new.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
