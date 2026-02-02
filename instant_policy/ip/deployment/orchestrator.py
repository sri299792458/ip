import pickle
from typing import Iterable, List, Optional

import numpy as np
import torch

from ip.deployment.config import DeploymentConfig
from ip.deployment.control.action_executor import ActionExecutor
from ip.deployment.control.ur_rtde_control import URRTDEControl
from ip.deployment.perception.realsense_perception import RealSensePerception
from ip.deployment.perception.sam_segmentation import build_segmenter
from ip.deployment.state.ur_rtde_state import URRTDEState
from ip.deployment.ur.robotiq_gripper import RobotiqGripper
from ip.models.diffusion import GraphDiffusion
from ip.utils.common_utils import transform_pcd
from ip.utils.data_proc import sample_to_cond_demo, save_sample, subsample_pcd


class InstantPolicyDeployment:
    def __init__(
        self,
        config: DeploymentConfig,
        rtde_control=None,
        rtde_receive=None,
        gripper=None,
        perception=None,
        state=None,
        control=None,
        load_model: bool = True,
        debug_gripper: bool = False,
    ):
        self.config = config
        self.perception = perception
        self.state = state
        self.control = control
        self._debug_gripper = debug_gripper

        if self.perception is None:
            if not config.camera_configs:
                raise ValueError("camera_configs must be provided when no perception instance is passed")
            segmenter = build_segmenter(
                config.segmentation,
                device=config.device,
                num_cameras=len(config.camera_configs),
            )
            self.perception = RealSensePerception(
                config.camera_configs,
                segmenter=segmenter,
                voxel_size=config.pcd_voxel_size,
            )

        if self.state is None or self.control is None:
            if gripper is None and config.gripper.enable:
                host = config.gripper.host or config.robot_ip
                gripper = RobotiqGripper(
                    host=host,
                    port=config.gripper.port,
                    open_position=config.gripper.open_position,
                    closed_position=config.gripper.closed_position,
                )
                gripper.connect()
                gripper.activate()
            if rtde_control is None:
                rtde_control = URRTDEControl.connect(config.robot_ip, config.rtde)
            if rtde_receive is None:
                rtde_receive = URRTDEState.connect(config.robot_ip)
            if self.state is None:
                self.state = URRTDEState(
                    rtde_receive,
                    gripper=gripper,
                    tcp_offset_in_code=config.tcp_offset_in_code,
                    tcp_offset_m=config.tcp_offset_m,
                )
        if self.control is None:
            self.control = URRTDEControl(
                rtde_control,
                control_config=config.rtde,
                gripper=gripper,
                gripper_config=config.gripper,
                tcp_offset_in_code=config.tcp_offset_in_code,
                tcp_offset_m=config.tcp_offset_m,
            )

        self.executor = ActionExecutor(self.control, self.state, config.safety, debug_gripper=debug_gripper)
        self.model = None
        self.model_config = None
        self._demo_embds = None
        self._demo_pos = None
        if load_model:
            self.model, self.model_config = self._load_model(
                config.model_path,
                config.num_demos,
                config.num_diffusion_iters,
                config.device,
            )

    def _load_model(self, model_path: str, num_demos: int, num_diffusion_iters: int, device: Optional[str]):
        config = pickle.load(open(f"{model_path}/config.pkl", "rb"))
        config["compile_models"] = False
        config["batch_size"] = 1
        config["num_demos"] = num_demos
        config["num_diffusion_iters_test"] = num_diffusion_iters
        if device:
            config["device"] = device

        model = GraphDiffusion.load_from_checkpoint(
            f"{model_path}/model.pt",
            config=config,
            strict=False,
            map_location=config["device"],
        ).to(config["device"])
        model.model.reinit_graphs(1, num_demos=max(num_demos, 1))
        model.eval()
        return model, config

    def _prepare_demos(self, demos: Iterable[dict]) -> List[dict]:
        prepared = []
        for demo in demos:
            if "obs" in demo:
                prepared.append(demo)
            else:
                prepared.append(
                    sample_to_cond_demo(demo, self.config.num_traj_wp, num_points=self.config.pcd_num_points)
                )
        if len(prepared) < self.model_config["num_demos"]:
            if not prepared:
                raise ValueError("At least one demo is required")
            while len(prepared) < self.model_config["num_demos"]:
                prepared.append(prepared[-1])
        return prepared[: self.model_config["num_demos"]]

    def run(self, demos: Iterable[dict], max_steps: Optional[int] = None, execution_horizon: Optional[int] = None) -> bool:
        if self.model is None or self.model_config is None:
            raise RuntimeError("Model is not loaded. Initialize with load_model=True to run deployment.")
        prepared_demos = self._prepare_demos(demos)
        pred_horizon = self.model_config["pre_horizon"]
        max_steps = max_steps or self.config.max_execution_steps
        execution_horizon = execution_horizon or pred_horizon

        full_sample = {"demos": prepared_demos, "live": {}}
        device = torch.device(self.model_config["device"])
        device_type = device.type

        for k in range(max_steps):
            T_w_e = self.state.get_T_w_e()
            grip_raw = self.state.get_gripper_state()
            grip = 1.0 if grip_raw >= 0.5 else 0.0  # 1=open, 0=closed
            pcd_w = self.perception.capture_pcd_world(use_segmentation=self.config.segmentation.enable)
            if self.config.show_masks:
                self._show_masks()
            if pcd_w.size == 0:
                print("Empty point cloud, skipping step")
                continue

            pcd_ee = transform_pcd(
                subsample_pcd(pcd_w, num_points=self.config.pcd_num_points),
                np.linalg.inv(T_w_e),
            )
            if self.config.debug_frame_sanity and (k % self.config.debug_frame_every == 0):
                self._print_frame_sanity(T_w_e, pcd_ee)

            full_sample["live"] = {
                "obs": [pcd_ee],
                "grips": [grip],
                "T_w_es": [T_w_e],
                "actions": [T_w_e.reshape(1, 4, 4).repeat(pred_horizon, axis=0)],
                "actions_grip": [np.zeros(pred_horizon)],
            }
            data = save_sample(full_sample, None)

            if k == 0:
                self._demo_embds, self._demo_pos = self.model.model.get_demo_scene_emb(data.to(device))

            data.live_scene_node_embds, data.live_scene_node_pos = self.model.model.get_live_scene_emb(data.to(device))
            data.demo_scene_node_embds = self._demo_embds.clone()
            data.demo_scene_node_pos = self._demo_pos.clone()

            with torch.no_grad():
                if device_type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float32):
                        actions, grips = self.model.test_step(data.to(device), 0)
                else:
                    actions, grips = self.model.test_step(data.to(device), 0)
                actions = actions.squeeze().cpu().numpy()
                grips = grips.squeeze().cpu().numpy()

            print("Policy output actions (first 3):")
            for idx in range(min(3, len(actions))):
                print(f"  [{idx}] T_e_e_rel:\n{actions[idx]}")
            state_label = "open" if grip >= 0.5 else "closed"
            print(f"Current gripper raw={grip_raw:.3f} bin={grip} ({state_label})")
            print("Policy output grips (first 8):", grips[: min(8, len(grips))])
            if self._debug_gripper:
                grip_cmds = (grips + 1.0) / 2.0
                grip_bins = (grip_cmds >= 0.5).astype(int)
                flips = int(np.sum(np.abs(np.diff(grip_bins[:pred_horizon]))))
                print("Policy grip cmds (first 8):", np.round(grip_cmds[: min(8, len(grip_cmds))], 3))
                print("Policy grip bins (first 8):", grip_bins[: min(8, len(grip_bins))].tolist())
                print(f"Pred horizon grip flips: {flips}")

            step_horizon = execution_horizon
            if step_horizon == pred_horizon and self.config.execute_until_grip_change:
                step_horizon = self._horizon_until_grip_change(grips, grip, pred_horizon)
            print(f"Step horizon: {step_horizon}/{pred_horizon} (execute_until_grip_change={self.config.execute_until_grip_change})")

            success, steps, error = self.executor.execute_actions(
                actions, grips, T_w_e, horizon=step_horizon
            )
            if not success:
                print(f"Execution failed at step {k}: {error}")
                return False

            print(f"Step {k}: executed {steps} actions")
        return True

    def _show_masks(self) -> None:
        try:
            import cv2
        except Exception:
            return

        frames = self.perception.get_last_debug_frames()
        for frame in frames:
            rgb = frame.get("rgb")
            if rgb is None:
                continue
            mask = frame.get("mask")
            overlay = rgb.copy()
            if mask is not None:
                if mask.shape != overlay.shape[:2]:
                    mask = None
                else:
                    green = np.zeros_like(overlay)
                    green[..., 1] = 255
                    overlay = np.where(
                        mask[..., None].astype(bool),
                        (0.3 * overlay + 0.7 * green).astype(overlay.dtype),
                        overlay,
                    )
            bgr = overlay[..., ::-1]
            cam_idx = frame.get("camera_index", 0)
            cv2.imshow(f"policy_masks_cam{cam_idx}", bgr)
        cv2.waitKey(1)

    def _print_frame_sanity(self, T_w_e: np.ndarray, pcd_ee: np.ndarray) -> None:
        print("Frame sanity:")
        if self.config.tcp_offset_in_code and self.config.tcp_offset_m is not None:
            offset = np.array(self.config.tcp_offset_m, dtype=np.float64).reshape(3)
            T_offset = np.eye(4, dtype=np.float64)
            T_offset[:3, 3] = offset
            T_w_flange = T_w_e @ np.linalg.inv(T_offset)
            delta = T_w_e[:3, 3] - T_w_flange[:3, 3]
            print(f"  tcp_offset_in_code=True, tcp_offset_m={offset.tolist()}")
            print(f"  T_w_e (tip, used by policy) pos: {np.round(T_w_e[:3, 3], 4)}")
            print(f"  T_w_flange (offset removed) pos: {np.round(T_w_flange[:3, 3], 4)}")
            print(
                "  tip-flange in world:",
                np.round(delta, 4),
                f"(norm {np.linalg.norm(delta):.4f} m)",
            )
            print("  Note: if robot TCP is already set to the tip, disable tcp_offset_in_code.")
        else:
            print("  tcp_offset_in_code=False; T_w_e is RTDE-reported TCP pose.")
            print(f"  T_w_e pos: {np.round(T_w_e[:3, 3], 4)}")

        if pcd_ee is not None and pcd_ee.size:
            mean = pcd_ee.mean(axis=0)
            p_min = pcd_ee.min(axis=0)
            p_max = pcd_ee.max(axis=0)
            print(f"  pcd_ee mean: {np.round(mean, 4)}")
            print(f"  pcd_ee bounds min: {np.round(p_min, 4)} max: {np.round(p_max, 4)}")

    @staticmethod
    def _horizon_until_grip_change(grips: np.ndarray, current_grip: float, max_horizon: int) -> int:
        current_state = 1.0 if current_grip >= 0.5 else 0.0
        grip_cmds = (grips[:max_horizon] + 1.0) / 2.0
        for i, cmd in enumerate(grip_cmds):
            next_state = 1.0 if cmd >= 0.5 else 0.0
            if next_state != current_state:
                return i + 1
        return max_horizon
