from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import threading
import time
from typing import List, Optional

import numpy as np

try:
    import serial
except Exception as exc:  # pragma: no cover - dependency may be missing in some environments
    serial = None
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None

from ip.deployment.spark_teleop.config import SparkDeviceConfig


@dataclass
class SparkPacket:
    timestamp: float
    device_id: str
    raw_values: List[int]
    status: List[bool]
    enable_switch: bool


@dataclass
class SparkSample:
    timestamp: float
    device_id: str
    angles_rad: List[float]
    raw_values: List[int]
    status: List[bool]
    enable_switch: bool


class SparkAngleUnwrapper:
    def __init__(self, offsets_raw: List[int], invert: List[int], modulus: int):
        if len(offsets_raw) != 7:
            raise ValueError(f"offsets_raw must have length 7, got {len(offsets_raw)}")
        if len(invert) != 7:
            raise ValueError(f"invert must have length 7, got {len(invert)}")
        self._prev = np.asarray(offsets_raw, dtype=np.int64)
        self._invert = np.asarray(invert, dtype=np.float64)
        self._modulus = int(modulus)
        self._angles = np.zeros(7, dtype=np.float64)

    def update(self, raw_values: List[int]) -> np.ndarray:
        cur = np.asarray(raw_values, dtype=np.int64)
        if cur.shape != (7,):
            raise ValueError(f"raw_values must have shape (7,), got {cur.shape}")
        dist = cur - self._prev
        half = self._modulus / 2
        dist = np.where(dist > half, dist - self._modulus, dist)
        dist = np.where(dist < -half, dist + self._modulus, dist)
        self._angles += self._invert * (dist.astype(np.float64) / float(self._modulus)) * (2.0 * np.pi)
        self._prev = cur
        return self._angles.copy()


class SparkSerialReader:
    def __init__(self, serial_device: str, baudrate: int, timeout_s: float):
        if serial is None:
            raise ImportError(
                "pyserial is required for Spark serial input. Install with `pip install pyserial`."
            ) from _SERIAL_IMPORT_ERROR
        self._serial_device = serial_device
        self._baudrate = baudrate
        self._timeout_s = timeout_s
        self._con = None

    def connect(self) -> None:
        if self._con is not None:
            return
        self._con = serial.Serial(self._serial_device, self._baudrate, timeout=self._timeout_s)
        time.sleep(0.5)
        self._con.reset_input_buffer()
        self._con.reset_output_buffer()
        self._con.read_until(b"\x00")

    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            finally:
                self._con = None

    def read_packet(self) -> Optional[SparkPacket]:
        if self._con is None:
            raise RuntimeError("SparkSerialReader is not connected.")
        payload = self._con.read_until(b"\x00")
        if not payload:
            return None
        data = payload[:-1] if payload.endswith(b"\x00") else payload
        if not data:
            return None
        parsed = json.loads(data.decode("utf-8"))
        raw_values = parsed.get("values", [])
        if len(raw_values) < 7:
            raise ValueError(f"Spark packet has {len(raw_values)} values; expected at least 7.")
        raw = [int(v) for v in raw_values[:7]]
        status = [bool(v) for v in parsed.get("status", [True] * 7)[:7]]
        return SparkPacket(
            timestamp=time.time(),
            device_id=str(parsed.get("ID", "")),
            raw_values=raw,
            status=status,
            enable_switch=bool(parsed.get("enable_switch", False)),
        )


def _load_offsets_pickle(path: str) -> tuple[List[int], Optional[List[int]]]:
    payload = pickle.load(Path(path).open("rb"))
    if isinstance(payload, tuple) and len(payload) >= 2:
        offsets_raw = [int(v) for v in payload[0][:7]]
        invert = [int(v) for v in payload[1][:7]]
        return offsets_raw, invert
    offsets_raw = [int(v) for v in payload[:7]]
    return offsets_raw, None


class SparkDevice:
    def __init__(self, config: SparkDeviceConfig):
        self.config = config
        self._reader = SparkSerialReader(
            serial_device=config.serial_device,
            baudrate=config.baudrate,
            timeout_s=config.timeout_s,
        )
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._latest = None
        self._unwrap = None
        self._last_error = None

    def _build_unwrapper(self, first_packet: SparkPacket) -> SparkAngleUnwrapper:
        offsets_raw = first_packet.raw_values
        invert = list(self.config.invert)
        if self.config.offsets_pickle:
            pickle_path = Path(self.config.offsets_pickle)
            if not pickle_path.exists():
                raise FileNotFoundError(
                    f"Offsets pickle does not exist for arm '{self.config.arm}': {pickle_path}"
                )
            file_offsets, file_invert = _load_offsets_pickle(str(pickle_path))
            offsets_raw = file_offsets
            if file_invert is not None:
                invert = file_invert
        return SparkAngleUnwrapper(
            offsets_raw=offsets_raw,
            invert=invert,
            modulus=self.config.encoder_modulus,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._reader.connect()
                packet = self._reader.read_packet()
                if packet is None:
                    continue
                if self._unwrap is None:
                    self._unwrap = self._build_unwrapper(packet)
                angles = self._unwrap.update(packet.raw_values).tolist()
                sample = SparkSample(
                    timestamp=packet.timestamp,
                    device_id=packet.device_id,
                    angles_rad=angles,
                    raw_values=list(packet.raw_values),
                    status=list(packet.status),
                    enable_switch=packet.enable_switch,
                )
                with self._lock:
                    self._latest = sample
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                try:
                    self._reader.close()
                except Exception:
                    pass
                time.sleep(0.2)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._reader.close()

    def get_latest(self) -> Optional[SparkSample]:
        with self._lock:
            return self._latest

    def get_last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error
