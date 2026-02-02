#!/usr/bin/env python3
"""
Print waypoint indices selected from a demo using extract_waypoints().
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from ip.utils.data_proc import extract_waypoints


def main():
    parser = argparse.ArgumentParser(description="Show selected waypoint indices from a demo.")
    parser.add_argument("--demo", required=True, help="Path to demo .pkl")
    parser.add_argument("--num-waypoints", type=int, default=10, help="Number of waypoints to select")
    args = parser.parse_args()

    demo_path = Path(args.demo)
    with demo_path.open("rb") as f:
        demo = pickle.load(f)

    T_w_es = demo.get("T_w_es", [])
    grips = demo.get("grips", [])
    if not T_w_es or not grips:
        raise RuntimeError("Demo must contain T_w_es and grips.")

    waypoints = extract_waypoints(np.array(T_w_es), np.array(grips), args.num_waypoints)
    print(f"Total frames: {len(T_w_es)}")
    print(f"Num waypoints: {args.num_waypoints}")
    print("Waypoint indices:", waypoints)


if __name__ == "__main__":
    main()
