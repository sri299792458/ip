import json
import pickle
import threading
import time
from pathlib import Path
from typing import Dict

import numpy as np

from ip.utils.data_proc import extract_waypoints, sample_to_cond_demo


class DemoCollector:
    def __init__(self, perception, state, control):
        self.perception = perception
        self.state = state
        self.control = control

    @staticmethod
    def _binarize_grips_from_obj(grip_objs) -> list:
        """Convert Robotiq OBJ status to binary grips.

        Standard Robotiq OBJ meanings:
          0 = moving
          1 = stopped on outer contact
          2 = stopped on inner contact
          3 = at requested position (no object)

        We treat OBJ in {1,2} as CLOSED (object contact).
        For OBJ in {0,3}, we treat the gripper as OPEN.
        No command/position-based fallback is used.
        """
        if not grip_objs or any(obj is None for obj in grip_objs):
            raise RuntimeError(
                "Robotiq OBJ feedback is required but missing. "
                "Ensure the gripper socket server exposes OBJ and try again."
            )
        grips = []
        for obj in grip_objs:
            if obj in {1, 2}:
                grips.append(0.0)
            else:
                grips.append(1.0)
        return grips

    def collect_kinesthetic(
        self,
        task_name: str,
        rate_hz: float = 10.0,
        use_segmentation: bool = False,
        debug_waypoints: bool = False,
    ) -> Dict:
        print(f"Collecting demo for: {task_name}")
        print("Move robot to start position, press ENTER to begin recording...")
        input()

        try:
            from pynput import keyboard
        except Exception as exc:
            raise ImportError("pynput is required for demo hotkeys. Install with `pip install pynput`.") from exc

        frames = {"pcds": [], "T_w_es": [], "grips": [], "grip_cmds": [], "grip_raws": [], "grip_objs": []}
        debug_rgb = [] if debug_waypoints else None
        stop_event = threading.Event()
        grip_cmd = {"value": None}
        missing_grip_reads = 0

        def on_press(key):
            if stop_event.is_set():
                return False
            if key == keyboard.Key.esc:
                stop_event.set()
                return False
            try:
                char = key.char.lower()
            except AttributeError:
                return None
            if char == "q":
                stop_event.set()
                return False
            if char == "o":
                if hasattr(self.control, "execute_gripper"):
                    self.control.execute_gripper(1.0)
                grip_cmd["value"] = 1.0
            elif char == "c":
                if hasattr(self.control, "execute_gripper"):
                    self.control.execute_gripper(0.0)
                grip_cmd["value"] = 0.0
            return None

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        print("Recording... hotkeys: o=open, c=close, q/esc=stop")
        if hasattr(self.control, "enable_freedrive"):
            self.control.enable_freedrive()

        try:
            period = 1.0 / rate_hz
            while not stop_event.is_set():
                start = time.time()
                pcd_w = self.perception.capture_pcd_world(use_segmentation=use_segmentation)
                if debug_waypoints:
                    debug_frames = self.perception.get_last_debug_frames()
                    rgb_pack = []
                    for frame in debug_frames:
                        rgb = frame.get("rgb")
                        if rgb is None:
                            continue
                        # Copy RGB buffers: RealSense frames can reuse underlying memory, so without
                        # a copy, many stored frames may end up identical.
                        rgb = np.asanyarray(rgb).copy()
                        entry = {"rgb": rgb}
                        if "serial" in frame:
                            entry["serial"] = frame["serial"]
                        elif "camera_index" in frame:
                            entry["serial"] = f"cam{frame['camera_index']}"
                        rgb_pack.append(entry)
                    debug_rgb.append(rgb_pack)
                T_w_e = self.state.get_T_w_e()
                grip_raw = self.state.get_gripper_state(default=None)
                grip_obj = None
                if hasattr(self.state, "_gripper") and self.state._gripper is not None:
                    try:
                        grip_obj = self.state._gripper.get_object_status()
                    except Exception:
                        grip_obj = None

                # Record measured gripper state (preferred) and also keep the user command for debugging.
                # Model convention: 1=open, 0=closed.
                if grip_raw is None:
                    missing_grip_reads += 1
                    if missing_grip_reads in {1, 50, 200}:
                        print(
                            "[warn] Gripper feedback read returned None. "
                            "This usually means GET POS failed; check Robotiq socket/URCap. "
                            "Continuing with last known state."
                        )
                    grip_raw_f = None
                else:
                    grip_raw_f = float(grip_raw)

                frames["pcds"].append(pcd_w)
                frames["T_w_es"].append(T_w_e)
                frames["grips"].append(None)
                cmd = grip_cmd["value"]
                frames["grip_cmds"].append(None if cmd is None else float(cmd))
                frames["grip_raws"].append(grip_raw_f)
                frames["grip_objs"].append(None if grip_obj is None else int(grip_obj))

                elapsed = time.time() - start
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            listener.stop()
            if hasattr(self.control, "disable_freedrive"):
                self.control.disable_freedrive()

        # Post-process measured gripper feedback into binary grips (measured-only).
        grips = self._binarize_grips_from_obj(frames["grip_objs"])
        frames["grips"] = grips

        print(f"Recorded {len(frames['pcds'])} frames")
        if debug_waypoints and debug_rgb is not None:
            frames["_debug_rgb"] = debug_rgb
        return frames

    def prepare_for_model(self, raw_demo: Dict, num_traj_wp: int = 10) -> Dict:
        return sample_to_cond_demo(raw_demo, num_traj_wp, require_grip_objs=True)

    def save_waypoint_debug_images(self, raw_demo: Dict, out_dir: str, num_traj_wp: int = 10) -> None:
        if "_debug_rgb" not in raw_demo:
            print("No debug RGB frames stored; re-run with debug_waypoints=True.")
            return
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("OpenCV is required to save waypoint debug images.") from exc

        T_w_es = raw_demo.get("T_w_es", [])
        grips = raw_demo.get("grips", [])
        if not T_w_es or not grips:
            print("No trajectory data found in demo.")
            return

        grip_objs = raw_demo.get("grip_objs", None)
        grip_objs = np.array(grip_objs) if grip_objs is not None else None
        waypoints = extract_waypoints(
            np.array(T_w_es),
            np.array(grips),
            num_waypoints=num_traj_wp,
            grip_objs=grip_objs,
            require_grip_objs=True,
        )

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with (out_path / "waypoints.json").open("w", encoding="utf-8") as f:
            payload = {
                "num_waypoints": int(num_traj_wp),
                # Cast to plain ints for JSON (may contain numpy scalars).
                "indices": [int(x) for x in waypoints],
                "waypoints": [],
            }
            for wp_i, frame_idx in enumerate(waypoints):
                entry = {
                    "wp": int(wp_i),
                    "frame": int(frame_idx),
                    "grip": None,
                    "grip_cmd": None,
                    "grip_raw": None,
                    "grip_obj": None,
                    "t_w_e_xyz": None,
                }
                if 0 <= int(frame_idx) < len(grips):
                    try:
                        entry["grip"] = float(grips[int(frame_idx)])
                    except Exception:
                        entry["grip"] = None
                grip_cmds = raw_demo.get("grip_cmds", [])
                if 0 <= int(frame_idx) < len(grip_cmds):
                    try:
                        entry["grip_cmd"] = None if grip_cmds[int(frame_idx)] is None else float(grip_cmds[int(frame_idx)])
                    except Exception:
                        entry["grip_cmd"] = None
                grip_raws = raw_demo.get("grip_raws", [])
                if 0 <= int(frame_idx) < len(grip_raws):
                    try:
                        entry["grip_raw"] = None if grip_raws[int(frame_idx)] is None else float(grip_raws[int(frame_idx)])
                    except Exception:
                        entry["grip_raw"] = None
                grip_objs = raw_demo.get("grip_objs", [])
                if 0 <= int(frame_idx) < len(grip_objs):
                    try:
                        entry["grip_obj"] = None if grip_objs[int(frame_idx)] is None else int(grip_objs[int(frame_idx)])
                    except Exception:
                        entry["grip_obj"] = None
                if 0 <= int(frame_idx) < len(T_w_es):
                    T = np.asarray(T_w_es[int(frame_idx)], dtype=np.float64)
                    if T.shape == (4, 4):
                        entry["t_w_e_xyz"] = [float(x) for x in T[:3, 3].tolist()]
                payload["waypoints"].append(entry)
            json.dump(payload, f, indent=2)

        rgb_frames = raw_demo["_debug_rgb"]
        for wp_i, frame_idx in enumerate(waypoints):
            if frame_idx < 0 or frame_idx >= len(rgb_frames):
                continue
            for cam_idx, cam in enumerate(rgb_frames[frame_idx]):
                rgb = cam.get("rgb")
                if rgb is None:
                    continue
                serial = cam.get("serial", f"cam{cam_idx}")
                safe_serial = "".join(
                    ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(serial)
                )
                filename = f"wp_{wp_i:02d}_frame_{frame_idx:04d}_{safe_serial}.png"
                bgr = rgb[:, :, ::-1] if rgb.ndim == 3 and rgb.shape[2] == 3 else rgb
                bgr = np.ascontiguousarray(bgr)
                bgr = bgr.copy()
                grip = raw_demo.get("grips", [None])[frame_idx] if raw_demo.get("grips") else None
                grip_cmd = raw_demo.get("grip_cmds", [None])[frame_idx] if raw_demo.get("grip_cmds") else None
                grip_raw = raw_demo.get("grip_raws", [None])[frame_idx] if raw_demo.get("grip_raws") else None
                grip_obj = raw_demo.get("grip_objs", [None])[frame_idx] if raw_demo.get("grip_objs") else None
                label = f"wp={wp_i} frame={frame_idx} grip={grip} cmd={grip_cmd} raw={grip_raw} obj={grip_obj}"
                cv2.putText(
                    bgr,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(out_path / filename), bgr)

    def save_demo(self, demo: Dict, path: str) -> None:
        parent = Path(path).expanduser().resolve().parent
        if parent.name:
            parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(demo, f)

    def load_demo(self, path: str) -> Dict:
        with open(path, "rb") as f:
            return pickle.load(f)
