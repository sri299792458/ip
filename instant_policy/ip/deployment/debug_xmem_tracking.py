#!/usr/bin/env python3
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - optional dependency
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

from ip.deployment.manual_seed_xmem import manual_seed_xmem
from ip.deployment.perception.realsense_perception import RealSensePerception
from ip.deployment.perception.sam_segmentation import build_segmenter


def _require_cv2():
    if cv2 is None:
        raise ImportError(f"OpenCV is required: {_CV2_IMPORT_ERROR}")


def _load_config():
    entry = Path(__file__).resolve().parents[1] / "deployment.py"
    spec = importlib.util.spec_from_file_location("ip_deploy_entry", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_default_config()


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.2).astype(np.uint8)
    return overlay


def main():
    _require_cv2()
    parser = argparse.ArgumentParser(description="Debug XMem++ tracking after manual seeding.")
    parser.add_argument("--frames", type=int, default=20, help="Frames to capture per camera")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between frames")
    parser.add_argument("--out-dir", default="ip/deployment/debug_outputs", help="Output folder for images")
    args = parser.parse_args()

    config = _load_config()
    if not config.segmentation.enable:
        raise RuntimeError("Segmentation is disabled in deployment.py")
    if config.segmentation.backend.lower() != "xmem":
        raise RuntimeError("XMem++ backend is required for this debug script")

    config.segmentation.xmem_init_with_sam = False
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segmenter = build_segmenter(
        config.segmentation,
        device=config.device,
        num_cameras=len(config.camera_configs),
    )
    perception = RealSensePerception(
        config.camera_configs,
        segmenter=segmenter,
        voxel_size=config.pcd_voxel_size,
    )

    try:
        manual_seed_xmem(
            perception,
            [cfg.serial for cfg in config.camera_configs],
            out_dir=str(out_dir),
        )

        for frame_idx in range(args.frames):
            for cam_idx, cam in enumerate(config.camera_configs):
                rgb = perception.capture_rgb(cam_idx, warmup=0)
                mask = segmenter.segment_camera(rgb, cam_idx)
                if mask is None:
                    print(f"[{cam.serial}] frame {frame_idx:03d}: mask None")
                    continue
                coverage = float(mask.sum()) / float(mask.size)
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                    print(
                        f"[{cam.serial}] frame {frame_idx:03d}: "
                        f"coverage {coverage:.4f}, centroid ({cx:.1f}, {cy:.1f})"
                    )
                else:
                    print(f"[{cam.serial}] frame {frame_idx:03d}: coverage {coverage:.4f}")

                overlay = _overlay_mask(rgb, mask)
                cv2.imwrite(str(out_dir / f"{cam.serial}_track_{frame_idx:03d}.png"), overlay)
                cv2.imwrite(str(out_dir / f"{cam.serial}_track_{frame_idx:03d}_mask.png"), mask * 255)

            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        perception.stop()

    print(f"Saved tracking overlays to {out_dir}")


if __name__ == "__main__":
    main()
