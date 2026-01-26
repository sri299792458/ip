import socket
import time
from typing import Optional


class RobotiqGripper:
    ACT = "ACT"
    GTO = "GTO"
    ATR = "ATR"
    ADR = "ADR"
    FOR = "FOR"
    SPE = "SPE"
    POS = "POS"
    STA = "STA"

    def __init__(
        self,
        host: str,
        port: int = 63352,
        socket_timeout: float = 2.0,
        open_position: int = 0,
        closed_position: int = 255,
    ):
        self._host = host
        self._port = port
        self._socket_timeout = socket_timeout
        self._socket = None
        self._open_position = open_position
        self._closed_position = closed_position
        self._last_position: Optional[int] = None

    @property
    def open_position(self) -> int:
        return self._open_position

    @property
    def closed_position(self) -> int:
        return self._closed_position

    def connect(self) -> None:
        if self._socket is not None:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._socket_timeout)
        self._socket.connect((self._host, self._port))

    def disconnect(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None

    def _set_var(self, variable: str, value: int) -> None:
        cmd = f"SET {variable} {value}\n"
        data = self._send_cmd(cmd)
        if data != b"ack":
            raise RuntimeError(f"Robotiq gripper did not ack {variable} set")

    def _get_var(self, variable: str) -> int:
        cmd = f"GET {variable}\n"
        data = self._send_cmd(cmd).decode("utf-8").strip()
        parts = data.split()
        if len(parts) != 2 or parts[0] != variable:
            raise RuntimeError(f"Unexpected gripper response: {data}")
        return int(parts[1])

    def _send_cmd(self, cmd: str) -> bytes:
        if self._socket is None:
            raise RuntimeError("Robotiq gripper is not connected")
        self._socket.sendall(cmd.encode("utf-8"))
        return self._socket.recv(1024)

    def activate(self) -> None:
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        time.sleep(0.5)
        self._set_var(self.ACT, 1)
        time.sleep(0.5)
        # STA can take a moment to reach 3 after activation.
        for _ in range(6):
            try:
                if self._get_var(self.STA) == 3:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("Robotiq gripper did not activate")

    def move(self, position: int, speed: int = 255, force: int = 100) -> None:
        position = int(max(0, min(255, position)))
        speed = int(max(0, min(255, speed)))
        force = int(max(0, min(255, force)))
        if self._last_position == position:
            return
        self._set_var(self.SPE, speed)
        self._set_var(self.FOR, force)
        self._set_var(self.POS, position)
        self._set_var(self.GTO, 1)
        self._last_position = position

    def open(self, speed: int = 255, force: int = 100) -> None:
        self.move(self._open_position, speed=speed, force=force)

    def close(self, speed: int = 255, force: int = 100) -> None:
        self.move(self._closed_position, speed=speed, force=force)

    def get_position(self) -> Optional[int]:
        try:
            return self._get_var(self.POS)
        except Exception:
            return None

    def get_position_normalized(self) -> Optional[float]:
        pos = self.get_position()
        if pos is None:
            return None
        denom = float(self._closed_position - self._open_position)
        if denom <= 0:
            return None
        return (pos - self._open_position) / denom
