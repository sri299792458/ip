from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

from ip.deployment.spark_teleop.config import (
    SparkTeleopConfig,
    default_spark_teleop_config,
    dump_spark_teleop_config,
    load_spark_teleop_config,
)


DEFAULT_CONFIG_PATH = Path("ip/deployment/spark_teleop/config.json")


def _load_or_default(path: Path) -> SparkTeleopConfig:
    if path.exists():
        return load_spark_teleop_config(path)
    cfg = default_spark_teleop_config()
    dump_spark_teleop_config(cfg, path, overwrite=False)
    print(f"[spark-teleop] Wrote default config to {path}")
    return cfg


def _filter_arms(config: SparkTeleopConfig, selected_arms: list[str]) -> SparkTeleopConfig:
    if not selected_arms:
        return config
    selected = {arm.lower() for arm in selected_arms}
    config.arms = {name: cfg for name, cfg in config.arms.items() if name.lower() in selected}
    config.spark_devices = [cfg for cfg in config.spark_devices if cfg.arm.lower() in selected]
    if not config.arms:
        raise ValueError(f"No matching arms after --arm filter: {selected_arms}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-Python SPARK teleop + demo collection (no ROS).")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Config path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write default config to --config and exit.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="Run only these arms (repeatable), e.g. --arm lightning --arm thunder",
    )
    parser.add_argument("--no-gui", action="store_true", help="Run without Tk GUI.")
    parser.add_argument(
        "--record-out",
        default=None,
        help="Optional output .pkl path for recording.",
    )
    parser.add_argument(
        "--lang-instruction",
        default="",
        help="Optional text stored in each recorded frame.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional run duration in seconds (useful with --no-gui).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.write_default_config:
        cfg = default_spark_teleop_config()
        dump_spark_teleop_config(cfg, config_path, overwrite=True)
        print(f"[spark-teleop] Wrote default config: {config_path}")
        return

    cfg = _load_or_default(config_path)
    cfg = _filter_arms(cfg, args.arm)

    from ip.deployment.spark_teleop.controller import SparkTeleopRuntime

    runtime = SparkTeleopRuntime(cfg)
    camera_manager = None
    recorder = None
    record_out_path = None
    try:
        runtime.start()
        print(f"[spark-teleop] Running arms: {runtime.get_arm_names()}")

        if args.record_out:
            from ip.deployment.spark_teleop.recorder import SparkDemoRecorder

            if cfg.recorder.include_cameras and cfg.cameras:
                from ip.deployment.spark_teleop.camera_io import CameraManager

                camera_manager = CameraManager(cfg.cameras)
            record_out_path = Path(args.record_out)
            if record_out_path.suffix == "":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                record_out_path = record_out_path / f"spark_demo_{timestamp}.pkl"
            recorder = SparkDemoRecorder(
                runtime=runtime,
                config=cfg,
                out_path=str(record_out_path),
                lang_instruction=args.lang_instruction,
                camera_manager=camera_manager,
            )
            recorder.start()
            print(f"[spark-teleop] Recording to: {record_out_path}")

        if args.no_gui:
            start = time.time()
            while True:
                if args.duration_s is not None and (time.time() - start) >= args.duration_s:
                    break
                time.sleep(0.1)
        else:
            from ip.deployment.spark_teleop.gui_tk import SparkTeleopGui

            gui = SparkTeleopGui(runtime, cfg)
            gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            stats = recorder.stop()
            print(
                f"[spark-teleop] Saved recording: {record_out_path} "
                f"(frames={stats.num_frames})"
            )
        if camera_manager is not None:
            camera_manager.stop()
        runtime.stop()
