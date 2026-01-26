#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

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

from ip.deployment.perception.sam_segmentation import build_segmenter


def _require_deps():
    if rs is None:
        raise ImportError(f"pyrealsense2 is required: {_RS_IMPORT_ERROR}")
    if cv2 is None:
        raise ImportError(f"OpenCV is required: {_CV2_IMPORT_ERROR}")


def _load_config():
    entry = Path(__file__).resolve().parents[1] / "deployment.py"
    spec = importlib.util.spec_from_file_location("ip_deploy_entry", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_default_config()


def _grab_color_frame(serial: str, width: int, height: int, fps: int, warmup: int = 30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    profile = pipeline.start(config)
    try:
        for _ in range(warmup):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"No color frame for serial {serial}")
        rgb = np.asanyarray(color_frame.get_data())
        return rgb
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def main():
    _require_deps()
    parser = argparse.ArgumentParser(description="Debug SAM/XMem segmentation masks per camera.")
    parser.add_argument("--out-dir", default="ip/deployment/debug_outputs", help="Output folder for images")
    args = parser.parse_args()

    config = _load_config()
    if not config.segmentation.enable:
        raise RuntimeError("Segmentation is disabled in deployment.py")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segmenter = build_segmenter(
        config.segmentation,
        device=config.device,
        num_cameras=len(config.camera_configs),
    )

    for idx, cam in enumerate(config.camera_configs):
        rgb = _grab_color_frame(cam.serial, cam.width, cam.height, cam.fps)
        mask = None

        # Inspect raw SAM masks if available (XMemOnlineSegmenter keeps a _sam field).
        sam = getattr(segmenter, "_sam", None)
        sam_generator = getattr(sam, "_generator", None)
        if sam_generator is not None:
            masks = sam_generator.generate(rgb)
            if masks:
                total = float(masks[0]["segmentation"].size)
                ratios = sorted(
                    [float(m.get("area", m["segmentation"].sum())) / total for m in masks],
                    reverse=True,
                )
                print(f"[{cam.serial}] SAM mask ratios (top 5): {[round(r, 4) for r in ratios[:5]]}")
                # Save the largest few SAM masks for inspection.
                for j, m in enumerate(sorted(masks, key=lambda x: x.get("area", 0), reverse=True)[:3]):
                    cv2.imwrite(
                        str(out_dir / f"{cam.serial}_sam_mask_{j}.png"),
                        (m["segmentation"].astype(np.uint8) * 255),
                    )

        if hasattr(segmenter, "segment_camera"):
            mask = segmenter.segment_camera(rgb, idx)
        else:
            mask = segmenter.segment(rgb)

        if mask is None:
            print(f"[{cam.serial}] mask: None")
            continue

        coverage = float(mask.sum()) / float(mask.size)
        print(f"[{cam.serial}] mask coverage: {coverage:.4f}")

        # Save color and mask for inspection.
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{cam.serial}_color.png"), rgb_bgr)
        cv2.imwrite(str(out_dir / f"{cam.serial}_mask.png"), (mask * 255).astype(np.uint8))

    print(f"Saved debug images to {out_dir}")


if __name__ == "__main__":
    main()
