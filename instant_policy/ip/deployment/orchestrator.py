import pickle
from typing import Iterable, List, Optional

import numpy as np
import torch

from ip.deployment.config import DeploymentConfig
from ip.deployment.control.action_executor import ActionExecutor
from ip.deployment.control.zeus_control import ZeusControl
from ip.deployment.perception.sam_segmentation import build_segmenter
from ip.deployment.perception.zeus_perception import ZeusPerception
from ip.deployment.state.zeus_state import ZeusState
from ip.deployment.zeus_env import ensure_zeus_on_path
from ip.models.diffusion import GraphDiffusion
from ip.utils.common_utils import transform_pcd
from ip.utils.data_proc import sample_to_cond_demo, save_sample, subsample_pcd


class InstantPolicyDeployment:
    def __init__(
        self,
        config: DeploymentConfig,
        move_group=None,
        perception=None,
        state=None,
        control=None,
    ):
        self.config = config
        self.perception = perception
        self.state = state
        self.control = control

        if self.perception is None:
            if not config.camera_configs:
                raise ValueError("camera_configs must be provided when no perception instance is passed")
            segmenter = build_segmenter(
                config.segmentation,
                device=config.device,
                num_cameras=len(config.camera_configs),
            )
            self.perception = ZeusPerception(
                config.camera_configs,
                segmenter=segmenter,
                voxel_size=config.pcd_voxel_size,
            )

        if self.state is None or self.control is None:
            if move_group is None:
                ensure_zeus_on_path()
                from common.move_group_interface import MoveGroupInterface
                enable_right = config.arm != "lightning"
                move_group = MoveGroupInterface(enable_right=enable_right)
            if self.state is None:
                self.state = ZeusState(move_group, arm=config.arm)
            if self.control is None:
                self.control = ZeusControl(move_group, arm=config.arm)

        self.executor = ActionExecutor(self.control, self.state, config.safety)
        self.model, self.model_config = self._load_model(
            config.model_path,
            config.num_demos,
            config.num_diffusion_iters,
            config.device,
        )
        self._demo_embds = None
        self._demo_pos = None

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
        prepared_demos = self._prepare_demos(demos)
        pred_horizon = self.model_config["pre_horizon"]
        max_steps = max_steps or self.config.max_execution_steps
        execution_horizon = execution_horizon or pred_horizon

        full_sample = {"demos": prepared_demos, "live": {}}
        device = torch.device(self.model_config["device"])
        device_type = device.type

        for k in range(max_steps):
            T_w_e = self.state.get_T_w_e()
            grip = self.state.get_gripper_state()
            grip = 1.0 if grip >= 0.5 else 0.0
            pcd_w = self.perception.capture_pcd_world(use_segmentation=self.config.segmentation.enable)
            if pcd_w.size == 0:
                print("Empty point cloud, skipping step")
                continue

            pcd_ee = transform_pcd(
                subsample_pcd(pcd_w, num_points=self.config.pcd_num_points),
                np.linalg.inv(T_w_e),
            )

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

            step_horizon = execution_horizon
            if step_horizon == pred_horizon and self.config.execute_until_grip_change:
                step_horizon = self._horizon_until_grip_change(grips, grip, pred_horizon)

            success, steps, error = self.executor.execute_actions(
                actions, grips, T_w_e, horizon=step_horizon
            )
            if not success:
                print(f"Execution failed at step {k}: {error}")
                return False

            print(f"Step {k}: executed {steps} actions")
        return True

    @staticmethod
    def _horizon_until_grip_change(grips: np.ndarray, current_grip: float, max_horizon: int) -> int:
        current_state = 1.0 if current_grip >= 0.5 else 0.0
        grip_cmds = (grips[:max_horizon] + 1.0) / 2.0
        for i, cmd in enumerate(grip_cmds):
            next_state = 1.0 if cmd >= 0.5 else 0.0
            if next_state != current_state:
                return i + 1
        return max_horizon
