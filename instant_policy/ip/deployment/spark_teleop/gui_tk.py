from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import tkinter as tk
from typing import Dict

import numpy as np

from ip.deployment.spark_teleop.config import GuiConfig, SparkTeleopConfig
from ip.deployment.spark_teleop.controller import ArmTelemetry, SparkTeleopRuntime
from ip.deployment.spark_teleop.kinematics import forward_xyz


@dataclass
class _ArmWidgets:
    status: tk.Label
    point_canvas: tk.Canvas
    point_id: int
    z_canvas: tk.Canvas
    z_id: int
    info: tk.Label
    enable_btn: tk.Button
    freedrive_btn: tk.Button
    freedrive_active: bool = False


class SparkTeleopGui:
    def __init__(self, runtime: SparkTeleopRuntime, config: SparkTeleopConfig):
        self.runtime = runtime
        self.config = config
        self.gui_cfg: GuiConfig = config.gui
        self._root = tk.Tk()
        self._root.title("SPARK Teleop (Pure Python)")
        self._root.geometry(f"{self.gui_cfg.window_width}x{self.gui_cfg.window_height}")
        self._widgets: Dict[str, _ArmWidgets] = {}
        self._build()

    def _build(self) -> None:
        arm_names = self.runtime.get_arm_names()
        for col, arm in enumerate(arm_names):
            frame = tk.Frame(self._root, borderwidth=2, relief=tk.GROOVE, padx=8, pady=8)
            frame.grid(row=0, column=col, sticky="nsew", padx=8, pady=8)

            self._root.grid_columnconfigure(col, weight=1)
            self._root.grid_rowconfigure(0, weight=1)

            title = tk.Label(frame, text=arm, font=("Helvetica", 22))
            title.grid(row=0, column=0, columnspan=5, sticky="w")

            status = tk.Label(frame, text="status: initializing", fg="black")
            status.grid(row=1, column=0, columnspan=5, sticky="w")

            enable_btn = tk.Button(frame, text="Disable", command=partial(self._toggle_enable, arm), width=12)
            enable_btn.grid(row=2, column=0)
            home_btn = tk.Button(frame, text="Home", command=partial(self._home, arm), width=12)
            home_btn.grid(row=2, column=1)
            gr_open_btn = tk.Button(frame, text="Gripper Open", command=partial(self._gripper_open, arm), width=12)
            gr_open_btn.grid(row=2, column=2)
            gr_close_btn = tk.Button(
                frame,
                text="Gripper Close",
                command=partial(self._gripper_close, arm),
                width=12,
            )
            gr_close_btn.grid(row=2, column=3)
            estop_btn = tk.Button(frame, text="E-Stop", command=partial(self._estop, arm), width=12, bg="#ff9999")
            estop_btn.grid(row=2, column=4)

            freedrive_btn = tk.Button(
                frame,
                text="Freedrive",
                command=partial(self._toggle_freedrive, arm),
                width=12,
            )
            freedrive_btn.grid(row=3, column=0)

            plot = tk.Canvas(frame, width=380, height=380, bg="white")
            plot.grid(row=4, column=0, columnspan=4, pady=8)
            cx, cy = 190, 190
            radius = int(self.gui_cfg.xy_bound_m * self.gui_cfg.xy_m_to_px)
            plot.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="black", width=2)
            point_id = plot.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="blue")

            zc = tk.Canvas(frame, width=60, height=380, bg="white")
            zc.grid(row=4, column=4, pady=8)
            z_bound_px = int(self.gui_cfg.z_bound_m * self.gui_cfg.z_m_to_px)
            zc.create_rectangle(8, 190 - z_bound_px, 52, 190 + z_bound_px, outline="black", width=2)
            z_id = zc.create_rectangle(10, 180, 50, 200, fill="blue")

            info = tk.Label(frame, text="dx=nan dy=nan dz=nan", justify=tk.LEFT)
            info.grid(row=5, column=0, columnspan=5, sticky="w")

            self._widgets[arm] = _ArmWidgets(
                status=status,
                point_canvas=plot,
                point_id=point_id,
                z_canvas=zc,
                z_id=z_id,
                info=info,
                enable_btn=enable_btn,
                freedrive_btn=freedrive_btn,
            )

    def _toggle_enable(self, arm: str) -> None:
        enabled = self.runtime.get_arm_enabled(arm)
        self.runtime.set_arm_enabled(arm, not enabled)
        self._widgets[arm].enable_btn.config(text=("Disable" if not enabled else "Enable"))

    def _home(self, arm: str) -> None:
        self.runtime.arm_runtimes[arm].move_home()

    def _gripper_open(self, arm: str) -> None:
        self.runtime.arm_runtimes[arm].open_gripper()

    def _gripper_close(self, arm: str) -> None:
        self.runtime.arm_runtimes[arm].close_gripper()

    def _estop(self, arm: str) -> None:
        self.runtime.arm_runtimes[arm].emergency_stop()

    def _toggle_freedrive(self, arm: str) -> None:
        w = self._widgets[arm]
        if not w.freedrive_active:
            self.runtime.arm_runtimes[arm].enable_freedrive()
            w.freedrive_active = True
            w.freedrive_btn.config(bg="#ffcc66")
        else:
            self.runtime.arm_runtimes[arm].disable_freedrive()
            w.freedrive_active = False
            w.freedrive_btn.config(bg="SystemButtonFace")

    def _compute_xyz_error(self, telem: ArmTelemetry) -> tuple[float, float, float]:
        if telem.robot_snapshot is None or telem.command_joints is None:
            return float("nan"), float("nan"), float("nan")
        ur_xyz = forward_xyz(np.asarray(telem.robot_snapshot.joint_positions, dtype=np.float64))
        spark_xyz = forward_xyz(np.asarray(telem.command_joints, dtype=np.float64))
        arm = telem.arm.lower()
        if arm == "thunder":
            x = -spark_xyz[1] + ur_xyz[1]
            y = -spark_xyz[0] + ur_xyz[0]
            z = spark_xyz[2] - ur_xyz[2]
        else:
            x = -((spark_xyz[1] - ur_xyz[1]))
            y = spark_xyz[0] - ur_xyz[0]
            z = -(spark_xyz[2] - ur_xyz[2])
        return float(x), float(y), float(z)

    def _refresh(self) -> None:
        snapshot = self.runtime.get_snapshot()
        cx, cy = 190, 190
        for arm, telem in snapshot.items():
            w = self._widgets[arm]
            dx, dy, dz = self._compute_xyz_error(telem)
            if np.isfinite([dx, dy, dz]).all():
                x_px = int(round(cx + dx * self.gui_cfg.xy_m_to_px))
                y_px = int(round(cy - dy * self.gui_cfg.xy_m_to_px))
                x_px = max(0, min(380, x_px))
                y_px = max(0, min(380, y_px))
                w.point_canvas.moveto(w.point_id, x_px - 8, y_px - 8)

                z_px = int(round(cy - dz * self.gui_cfg.z_m_to_px))
                z_px = max(0, min(380, z_px))
                w.z_canvas.coords(w.z_id, 10, z_px - 10, 50, z_px + 10)

                in_bounds = (
                    abs(dx) <= self.gui_cfg.xy_bound_m
                    and abs(dy) <= self.gui_cfg.xy_bound_m
                    and abs(dz) <= self.gui_cfg.z_bound_m
                )
                color = "#3cb371" if in_bounds else "#ff8c00"
                w.point_canvas.itemconfig(w.point_id, fill=color)
                w.z_canvas.itemconfig(w.z_id, fill=color)
                w.status.config(
                    text=(
                        f"status: {'enabled' if telem.enabled else 'disabled'} | "
                        f"spark_enable={telem.spark_sample.enable_switch if telem.spark_sample else None} | "
                        f"{'stale' if telem.stale else 'live'}"
                    ),
                    fg="black" if in_bounds else "#cc5500",
                )
                w.info.config(text=f"dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} m")
            else:
                w.status.config(
                    text=f"status: waiting ({telem.last_error or 'no data'})",
                    fg="#7a7a7a",
                )
                w.info.config(text="dx=nan dy=nan dz=nan")
                w.point_canvas.itemconfig(w.point_id, fill="blue")
                w.z_canvas.itemconfig(w.z_id, fill="blue")

        delay_ms = int(round(1000.0 / max(self.gui_cfg.refresh_hz, 1e-3)))
        self._root.after(delay_ms, self._refresh)

    def run(self) -> None:
        self._refresh()
        self._root.mainloop()
