from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List


THUNDER_OFFSET_RAD = [
    0.8322513103485107,
    1.3889789581298828,
    1.4154774993658066,
    -2.7204548865556717,
    -2.634120313450694,
    -2.2259570360183716,
    0.0,
]

LIGHTNING_OFFSET_RAD = [
    -1.0215797424316406,
    -4.490872740745544,
    -1.4827108010649681,
    -0.588315486907959,
    -0.5356001891195774,
    2.0629922747612,
    0.0,
]


def _discover_offsets_pickle(filename: str) -> str:
    """Best-effort discovery for local SPARK offsets pickle files."""
    module_path = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in module_path.parents:
        candidates.append(parent / "SPARK-Remote-data_collection" / "TeleopSoftware" / "Spark" / filename)
    candidates.append(Path.cwd() / "SPARK-Remote-data_collection" / "TeleopSoftware" / "Spark" / filename)
    candidates.append(Path.cwd() / ".." / "SPARK-Remote-data_collection" / "TeleopSoftware" / "Spark" / filename)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return str(resolved)
    return ""


def _assert_len(values: List[float], expected_len: int, field_name: str) -> None:
    if len(values) != expected_len:
        raise ValueError(f"{field_name} must have length {expected_len}, got {len(values)}")


@dataclass
class SparkDeviceConfig:
    arm: str
    serial_device: str
    offsets_pickle: str = ""
    invert: List[int] = field(default_factory=lambda: [-1, -1, 1, -1, -1, -1, -1])
    baudrate: int = 921600
    timeout_s: float = 0.2
    encoder_modulus: int = 16384

    @staticmethod
    def from_dict(data: dict) -> "SparkDeviceConfig":
        cfg = SparkDeviceConfig(
            arm=str(data["arm"]),
            serial_device=str(data["serial_device"]),
            offsets_pickle=str(data.get("offsets_pickle", "")),
            invert=[int(v) for v in data.get("invert", [-1, -1, 1, -1, -1, -1, -1])],
            baudrate=int(data.get("baudrate", 921600)),
            timeout_s=float(data.get("timeout_s", 0.2)),
            encoder_modulus=int(data.get("encoder_modulus", 16384)),
        )
        _assert_len(cfg.invert, 7, "SparkDeviceConfig.invert")
        return cfg


@dataclass
class GripperMapConfig:
    raw_min: float
    raw_max: float
    open_position: int = 0
    closed_position: int = 255

    @staticmethod
    def from_dict(data: dict) -> "GripperMapConfig":
        return GripperMapConfig(
            raw_min=float(data["raw_min"]),
            raw_max=float(data["raw_max"]),
            open_position=int(data.get("open_position", 0)),
            closed_position=int(data.get("closed_position", 255)),
        )


@dataclass
class ArmControlConfig:
    command_rate_hz: float = 200.0
    servo_dt_s: float = 0.001
    servo_lookahead_s: float = 0.05
    servo_gain: int = 200

    @staticmethod
    def from_dict(data: dict) -> "ArmControlConfig":
        return ArmControlConfig(
            command_rate_hz=float(data.get("command_rate_hz", 200.0)),
            servo_dt_s=float(data.get("servo_dt_s", 0.001)),
            servo_lookahead_s=float(data.get("servo_lookahead_s", 0.05)),
            servo_gain=int(data.get("servo_gain", 200)),
        )


@dataclass
class ArmConfig:
    name: str
    robot_ip: str
    enabled: bool = True
    use_gripper: bool = True
    home_joint_rad: List[float] = field(default_factory=lambda: [0.0] * 6)
    spark_joint_offset_rad: List[float] = field(default_factory=lambda: [0.0] * 7)
    gripper_map: GripperMapConfig = field(
        default_factory=lambda: GripperMapConfig(raw_min=-4.8, raw_max=-2.3)
    )
    control: ArmControlConfig = field(default_factory=ArmControlConfig)
    stale_timeout_s: float = 0.25

    @staticmethod
    def from_dict(name: str, data: dict) -> "ArmConfig":
        cfg = ArmConfig(
            name=name,
            robot_ip=str(data["robot_ip"]),
            enabled=bool(data.get("enabled", True)),
            use_gripper=bool(data.get("use_gripper", True)),
            home_joint_rad=[float(v) for v in data.get("home_joint_rad", [0.0] * 6)],
            spark_joint_offset_rad=[
                float(v) for v in data.get("spark_joint_offset_rad", [0.0] * 7)
            ],
            gripper_map=GripperMapConfig.from_dict(
                data.get(
                    "gripper_map",
                    {"raw_min": -4.8, "raw_max": -2.3, "open_position": 0, "closed_position": 255},
                )
            ),
            control=ArmControlConfig.from_dict(data.get("control", {})),
            stale_timeout_s=float(data.get("stale_timeout_s", 0.25)),
        )
        _assert_len(cfg.home_joint_rad, 6, f"ArmConfig[{name}].home_joint_rad")
        _assert_len(cfg.spark_joint_offset_rad, 7, f"ArmConfig[{name}].spark_joint_offset_rad")
        return cfg


@dataclass
class CameraStreamConfig:
    role: str
    serial: str
    enabled: bool = True
    width: int = 640
    height: int = 480
    fps: int = 30

    @staticmethod
    def from_dict(data: dict) -> "CameraStreamConfig":
        return CameraStreamConfig(
            role=str(data["role"]),
            serial=str(data["serial"]),
            enabled=bool(data.get("enabled", True)),
            width=int(data.get("width", 640)),
            height=int(data.get("height", 480)),
            fps=int(data.get("fps", 30)),
        )


@dataclass
class RecorderConfig:
    enabled: bool = False
    frame_rate_hz: float = 15.0
    out_dir: str = "demos/spark"
    include_cameras: bool = True
    include_depth: bool = True

    @staticmethod
    def from_dict(data: dict) -> "RecorderConfig":
        return RecorderConfig(
            enabled=bool(data.get("enabled", False)),
            frame_rate_hz=float(data.get("frame_rate_hz", 15.0)),
            out_dir=str(data.get("out_dir", "demos/spark")),
            include_cameras=bool(data.get("include_cameras", True)),
            include_depth=bool(data.get("include_depth", True)),
        )


@dataclass
class GuiConfig:
    enabled: bool = True
    refresh_hz: float = 30.0
    window_width: int = 1400
    window_height: int = 850
    xy_m_to_px: float = 400.0
    z_m_to_px: float = 350.0
    xy_bound_m: float = 0.12
    z_bound_m: float = 0.12

    @staticmethod
    def from_dict(data: dict) -> "GuiConfig":
        return GuiConfig(
            enabled=bool(data.get("enabled", True)),
            refresh_hz=float(data.get("refresh_hz", 30.0)),
            window_width=int(data.get("window_width", 1400)),
            window_height=int(data.get("window_height", 850)),
            xy_m_to_px=float(data.get("xy_m_to_px", 400.0)),
            z_m_to_px=float(data.get("z_m_to_px", 350.0)),
            xy_bound_m=float(data.get("xy_bound_m", 0.12)),
            z_bound_m=float(data.get("z_bound_m", 0.12)),
        )


@dataclass
class SparkTeleopConfig:
    arms: Dict[str, ArmConfig] = field(default_factory=dict)
    spark_devices: List[SparkDeviceConfig] = field(default_factory=list)
    cameras: List[CameraStreamConfig] = field(default_factory=list)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    stop_all_on_fault: bool = True

    @staticmethod
    def from_dict(data: dict) -> "SparkTeleopConfig":
        arm_dict = data.get("arms", {})
        arms = {
            arm_name: ArmConfig.from_dict(arm_name, arm_data)
            for arm_name, arm_data in arm_dict.items()
        }
        spark_devices = [SparkDeviceConfig.from_dict(x) for x in data.get("spark_devices", [])]
        cameras = [CameraStreamConfig.from_dict(x) for x in data.get("cameras", [])]
        return SparkTeleopConfig(
            arms=arms,
            spark_devices=spark_devices,
            cameras=cameras,
            recorder=RecorderConfig.from_dict(data.get("recorder", {})),
            gui=GuiConfig.from_dict(data.get("gui", {})),
            stop_all_on_fault=bool(data.get("stop_all_on_fault", True)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def default_spark_teleop_config() -> SparkTeleopConfig:
    lightning_offsets = _discover_offsets_pickle("offsets_lightning.pickle")
    thunder_offsets = _discover_offsets_pickle("offsets_thunder.pickle")
    return SparkTeleopConfig(
        arms={
            "lightning": ArmConfig(
                name="lightning",
                robot_ip="10.33.55.90",
                spark_joint_offset_rad=list(LIGHTNING_OFFSET_RAD),
                gripper_map=GripperMapConfig(raw_min=-4.8, raw_max=-2.3),
            ),
            "thunder": ArmConfig(
                name="thunder",
                robot_ip="10.33.55.89",
                spark_joint_offset_rad=list(THUNDER_OFFSET_RAD),
                gripper_map=GripperMapConfig(raw_min=-2.71, raw_max=-1.26),
            ),
        },
        spark_devices=[
            SparkDeviceConfig(
                arm="lightning",
                serial_device="/dev/ttyUSB0",
                offsets_pickle=lightning_offsets,
                invert=[-1, -1, 1, -1, -1, -1, -1],
            ),
            SparkDeviceConfig(
                arm="thunder",
                serial_device="/dev/ttyUSB1",
                offsets_pickle=thunder_offsets,
                invert=[-1, -1, 1, -1, -1, -1, 1],
            ),
        ],
        cameras=[],
        recorder=RecorderConfig(enabled=False),
        gui=GuiConfig(enabled=True),
        stop_all_on_fault=True,
    )


def load_spark_teleop_config(path: str | Path) -> SparkTeleopConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SparkTeleopConfig.from_dict(data)


def dump_spark_teleop_config(
    config: SparkTeleopConfig,
    path: str | Path,
    overwrite: bool = False,
) -> Path:
    config_path = Path(path)
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    return config_path
