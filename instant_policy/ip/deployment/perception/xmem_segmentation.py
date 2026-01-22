import os
import sys
import time
from typing import List, Optional

import numpy as np

try:
    import rospy
    from sensor_msgs.msg import Image
except Exception as exc:  # pragma: no cover - ROS runtime dependency
    rospy = None
    Image = None
    _ROS_IMPORT_ERROR = exc
else:
    _ROS_IMPORT_ERROR = None


def _ensure_xmem_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    xmem_path = os.path.join(repo_root, "XMem2-main")
    if xmem_path not in sys.path:
        sys.path.insert(0, xmem_path)
    return xmem_path


class XMemMaskSubscriber:
    def __init__(self, mask_topics: List[str], timeout_s: float = 0.5, threshold: float = 0.5):
        if rospy is None:
            raise ImportError(f"rospy is required for XMem masks: {_ROS_IMPORT_ERROR}")
        if not mask_topics:
            raise ValueError("mask_topics must be provided for XMem mask subscription")

        self._timeout_s = timeout_s
        self._threshold = threshold
        self._masks = [None for _ in mask_topics]
        self._stamps = [None for _ in mask_topics]

        for idx, topic in enumerate(mask_topics):
            rospy.Subscriber(topic, Image, self._on_mask, callback_args=idx, queue_size=1)

    def _on_mask(self, msg: Image, idx: int):
        mask = self._image_to_mask(msg)
        self._masks[idx] = mask
        self._stamps[idx] = time.time()

    def _image_to_mask(self, msg: Image) -> np.ndarray:
        h, w = msg.height, msg.width
        if msg.encoding in ("mono8", "8UC1"):
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
            mask = data > 0
        elif msg.encoding in ("16UC1",):
            data = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
            mask = data > 0
        elif msg.encoding in ("32FC1",):
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
            mask = data > self._threshold
        else:
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
            mask = data > 0
        return mask.astype(np.uint8)

    def get_masks(self) -> List[Optional[np.ndarray]]:
        now = time.time()
        masks = []
        for mask, stamp in zip(self._masks, self._stamps):
            if mask is None or stamp is None or (now - stamp) > self._timeout_s:
                masks.append(None)
            else:
                masks.append(mask)
        return masks


class XMemOnlineSegmenter:
    def __init__(
        self,
        num_cameras: int,
        checkpoint_path: Optional[str],
        device: Optional[str] = None,
        init_with_sam: bool = True,
        sam_config=None,
        config_overrides: Optional[dict] = None,
    ):
        _ensure_xmem_on_path()
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        device_str = str(device)
        if not device_str.startswith("cuda"):
            raise RuntimeError("XMem++ requires CUDA for real-time use")
        if device_str not in ("cuda", "cuda:0"):
            raise RuntimeError("XMem++ inference expects device cuda:0")
        if device_str == "cuda":
            device = "cuda:0"

        from model.network import XMem
        from util.configuration import VIDEO_INFERENCE_CONFIG
        from dataset.range_transform import im_normalization
        import torch

        if not checkpoint_path:
            raise ValueError("XMem++ checkpoint_path is required")

        self._device = torch.device(device)
        self._im_normalization = im_normalization
        self._num_cameras = num_cameras
        self._initialized = [False] * num_cameras

        config = VIDEO_INFERENCE_CONFIG.copy()
        config["model"] = checkpoint_path
        config["size"] = -1
        config["save_masks"] = False
        if config_overrides:
            config.update(config_overrides)
        self._config = config

        self._network = XMem(
            config,
            checkpoint_path,
            pretrained_key_encoder=False,
            pretrained_value_encoder=False,
        ).to(self._device).eval()

        from inference.inference_core import InferenceCore
        self._processors = [InferenceCore(self._network, config) for _ in range(num_cameras)]
        self._labels = [1]

        self._sam = None
        if init_with_sam:
            if sam_config is None:
                raise ValueError("SAM config is required to seed XMem++")
            from ip.deployment.perception.sam_segmentation import SAMSegmenter
            checkpoint = sam_config.sam_checkpoint_path or sam_config.checkpoint_path
            if not checkpoint:
                raise ValueError("SAM checkpoint is required to seed XMem++")
            self._sam = SAMSegmenter(
                model_type=sam_config.model_type,
                checkpoint_path=checkpoint,
                device=device,
                points_per_side=sam_config.points_per_side,
                pred_iou_thresh=sam_config.pred_iou_thresh,
                stability_score_thresh=sam_config.stability_score_thresh,
                min_mask_region_area=sam_config.min_mask_region_area,
                select_largest=sam_config.select_largest,
            )

        self._torch = torch

    def segment_camera(self, rgb: np.ndarray, camera_index: int) -> Optional[np.ndarray]:
        if camera_index >= self._num_cameras:
            return None

        if not self._initialized[camera_index]:
            if self._sam is None:
                return None
            mask = self._sam.segment(rgb)
            if mask is None or mask.sum() == 0:
                return None
            self._initialize(camera_index, rgb, mask)
            return mask.astype(np.uint8)

        return self._track(camera_index, rgb)

    def initialize_camera(self, camera_index: int, rgb: np.ndarray, mask: np.ndarray) -> None:
        if camera_index >= self._num_cameras:
            return
        self._initialize(camera_index, rgb, mask)

    def _initialize(self, camera_index: int, rgb: np.ndarray, mask: np.ndarray) -> None:
        processor = self._processors[camera_index]
        processor.clear_memory()
        processor.set_all_labels(self._labels)

        image_t = self._prepare_image(rgb)
        mask_t = self._prepare_mask(mask)
        with self._torch.no_grad():
            processor.put_to_permanent_memory(image_t, mask_t, ti=0)
        self._initialized[camera_index] = True

    def _track(self, camera_index: int, rgb: np.ndarray) -> Optional[np.ndarray]:
        processor = self._processors[camera_index]
        image_t = self._prepare_image(rgb)
        with self._torch.no_grad():
            prob = processor.step(image_t, mask=None, valid_labels=None)
        if prob is None:
            return None
        pred = self._torch.argmax(prob, dim=0).detach().cpu().numpy().astype(np.uint8)
        return (pred > 0).astype(np.uint8)

    def _prepare_image(self, rgb: np.ndarray):
        image = self._torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        image = self._im_normalization(image)
        return image.to(self._device)

    def _prepare_mask(self, mask: np.ndarray):
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = (mask > 0).astype(np.float32)
        mask_t = self._torch.from_numpy(mask)
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)
        return mask_t.to(self._device)
