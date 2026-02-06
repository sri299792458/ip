# SPARK Teleop (Pure Python, No ROS)

This guide covers the new ROS-free SPARK teleop runtime inside `ip/deployment/spark_teleop`.

## Scope
- SPARK serial input (`pyserial`)
- UR5e command/state (`ur_rtde`)
- Robotiq gripper socket control
- Optional RealSense recording
- Tk GUI for Spark-vs-UR bound monitoring

## Generate Config
```bash
python -m ip.deployment.spark_teleop --write-default-config \
  --config ip/deployment/spark_teleop/config.json
```

Install deployment extras first:
```bash
pip install -e ".[deployment]"
```

Then edit:
- `arms.<arm>.robot_ip`
- `spark_devices[*].serial_device`
- `spark_devices[*].offsets_pickle` (if using existing SPARK offsets)
- optional `cameras` entries for recording

Offsets behavior:
- If SPARK offsets pickle is discovered locally, defaults are pre-filled.
- If `offsets_pickle` is set but the file is missing, teleop fails fast.
- If `offsets_pickle` is empty, first packet raw angles are used as startup offsets.

## Run Teleop
```bash
python -m ip.deployment.spark_teleop --config ip/deployment/spark_teleop/config.json
```

Single arm:
```bash
python -m ip.deployment.spark_teleop --config ip/deployment/spark_teleop/config.json --arm lightning
```

Headless:
```bash
python -m ip.deployment.spark_teleop --config ip/deployment/spark_teleop/config.json --no-gui
```

## Record Demo
```bash
python -m ip.deployment.spark_teleop \
  --config ip/deployment/spark_teleop/config.json \
  --record-out demos/spark/demo1.pkl \
  --lang-instruction "pick the blue block into the black bowl"
```

Output schema:
- `meta`
  - `schema_version = "spark_teleop_v1"`
  - `recorded_at_utc`, `arms`, `lang_instruction`, `num_frames`
- `frames[i]`
  - `timestamp`
  - `arms.<arm>`:
    - spark angles/raw/enable
    - commanded joints + gripper closed value
    - measured joints/tcp/ft/gripper
  - optional `cameras`

## Notes
- This stack is independent of ROS1/ROS2.
- It is bimanual-capable, with single-arm operation controlled by `--arm`.
- While Freedrive is active on an arm, Spark servoJ streaming for that arm is paused.
