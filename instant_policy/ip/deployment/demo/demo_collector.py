import pickle
import threading
import time
from typing import Dict

from ip.utils.data_proc import sample_to_cond_demo


class DemoCollector:
    def __init__(self, perception, state, control):
        self.perception = perception
        self.state = state
        self.control = control

    def collect_kinesthetic(self, task_name: str, rate_hz: float = 10.0, use_segmentation: bool = False) -> Dict:
        print(f"Collecting demo for: {task_name}")
        print("Move robot to start position, press ENTER to begin recording...")
        input()

        if hasattr(self.control, "enable_freedrive"):
            self.control.enable_freedrive()

        frames = {"pcds": [], "T_w_es": [], "grips": []}
        stop_event = threading.Event()

        def wait_stop():
            input("Recording... Press ENTER to stop.\n")
            stop_event.set()

        threading.Thread(target=wait_stop, daemon=True).start()

        period = 1.0 / rate_hz
        while not stop_event.is_set():
            start = time.time()
            pcd_w = self.perception.capture_pcd_world(use_segmentation=use_segmentation)
            T_w_e = self.state.get_T_w_e()
            grip = self.state.get_gripper_state()
            grip = 1.0 if grip >= 0.5 else 0.0

            frames["pcds"].append(pcd_w)
            frames["T_w_es"].append(T_w_e)
            frames["grips"].append(grip)

            elapsed = time.time() - start
            if elapsed < period:
                time.sleep(period - elapsed)

        if hasattr(self.control, "disable_freedrive"):
            self.control.disable_freedrive()

        print(f"Recorded {len(frames['pcds'])} frames")
        return frames

    def prepare_for_model(self, raw_demo: Dict, num_traj_wp: int = 10) -> Dict:
        return sample_to_cond_demo(raw_demo, num_traj_wp)

    def save_demo(self, demo: Dict, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(demo, f)

    def load_demo(self, path: str) -> Dict:
        with open(path, "rb") as f:
            return pickle.load(f)
