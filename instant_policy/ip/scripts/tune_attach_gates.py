import argparse
import csv
import itertools
import json
from typing import Dict, List

import numpy as np

from ip.generation.config import GenerationConfig
from ip.generation.pseudo_demo_generator import PseudoDemoGenerator


DEFAULT_SKILLS = ["random", "grasp", "pick_place", "pull", "push"]


def _new_totals() -> Dict[str, float]:
    return {
        "num_tasks": 0.0,
        "num_demos": 0.0,
        "close_events": 0.0,
        "target_close_events": 0.0,
        "untargeted_close_events": 0.0,
        "attach_success": 0.0,
        "attach_success_target": 0.0,
        "attach_success_untargeted": 0.0,
        "attach_miss_target": 0.0,
        "attach_miss_untargeted": 0.0,
        "detach_events": 0.0,
        "dist_sum_attached": 0.0,
        "dist_count_attached": 0.0,
        "cap_sum_attached": 0.0,
        "dist_sum_failed_target": 0.0,
        "dist_count_failed_target": 0.0,
        "cap_sum_failed_target": 0.0,
        "hard_negative_probes": 0.0,
        "hard_negative_false_attach": 0.0,
    }


def _acc(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for key, val in src.items():
        dst[key] = float(dst.get(key, 0.0)) + float(val)


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _summarize(totals: Dict[str, float]) -> Dict[str, float]:
    close_events = totals["close_events"]
    target_close = totals["target_close_events"]
    untarget_close = totals["untargeted_close_events"]

    attach_rate = _safe_div(totals["attach_success"], close_events)
    target_attach_rate = _safe_div(totals["attach_success_target"], target_close)
    untarget_attach_rate = _safe_div(totals["attach_success_untargeted"], untarget_close)
    target_miss_rate = _safe_div(totals["attach_miss_target"], target_close)
    untarget_miss_rate = _safe_div(totals["attach_miss_untargeted"], untarget_close)

    mean_dist_attached = _safe_div(totals["dist_sum_attached"], totals["dist_count_attached"])
    mean_cap_attached = _safe_div(totals["cap_sum_attached"], totals["attach_success"])
    mean_dist_failed_target = _safe_div(
        totals["dist_sum_failed_target"], totals["dist_count_failed_target"]
    )
    mean_cap_failed_target = _safe_div(
        totals["cap_sum_failed_target"], totals["attach_miss_target"]
    )
    hard_negative_false_rate = _safe_div(
        totals["hard_negative_false_attach"], totals["hard_negative_probes"]
    )

    # Rank by attach recall while penalizing false attach on hard-negative probes.
    score = 0.6 * target_attach_rate + 0.2 * attach_rate + 0.2 * (1.0 - hard_negative_false_rate)

    return {
        "score": score,
        "attach_rate": attach_rate,
        "target_attach_rate": target_attach_rate,
        "untarget_attach_rate": untarget_attach_rate,
        "target_miss_rate": target_miss_rate,
        "untarget_miss_rate": untarget_miss_rate,
        "mean_dist_attached": mean_dist_attached,
        "mean_cap_attached": mean_cap_attached,
        "mean_dist_failed_target": mean_dist_failed_target,
        "mean_cap_failed_target": mean_cap_failed_target,
        "hard_negative_false_rate": hard_negative_false_rate,
    }


def _build_config(args, skill: str, attach_radius: float, capture_min_points: int) -> GenerationConfig:
    return GenerationConfig(
        shapenet_path=args.shapenet_path,
        shapenet_index_path=args.shapenet_index_path,
        save_dir=args.save_dir,
        num_tasks=args.num_tasks,
        num_demos_per_task=(args.num_demos_per_task, args.num_demos_per_task),
        bias_prob=args.bias_prob,
        forced_skill=skill,
        num_objects_range=tuple(args.num_objects_range),
        num_waypoints_range=tuple(args.num_waypoints_range),
        object_scale_range=tuple(args.object_scale_range),
        trans_spacing=args.trans_spacing,
        rot_spacing_deg=args.rot_spacing_deg,
        interpolation_methods=tuple(args.interpolation_methods),
        disturbance_prob=args.disturbance_prob,
        gripper_noise_prob=args.gripper_noise_prob,
        attach_on_grasp=not args.no_attach,
        attach_radius=float(attach_radius),
        attach_capture_min_points=int(capture_min_points),
        gripper_mesh_path=args.gripper_mesh_path,
        max_meshes=args.max_meshes,
        cache_meshes=args.cache_meshes,
        seed=args.seed,
        save_renders=False,
        render_make_videos=False,
    )


def _make_hard_negative_local_offsets(offset_m: float) -> np.ndarray:
    d = float(offset_m)
    return np.array(
        [
            [d, 0.0, 0.0],
            [-d, 0.0, 0.0],
            [0.0, d, 0.0],
            [0.0, -d, 0.0],
            [0.0, 0.0, d],
            [0.0, 0.0, -d],
        ],
        dtype=np.float64,
    )


def _run_combo(args, attach_radius: float, capture_min_points: int, skills: List[str]):
    combo_totals = _new_totals()
    per_skill = {}
    hard_negative_local_offsets = _make_hard_negative_local_offsets(args.hard_negative_offset_m)

    for skill_idx, skill in enumerate(skills):
        config = _build_config(args, skill, attach_radius, capture_min_points)
        generator = PseudoDemoGenerator(config, scene_encoder=None, build_renderer=False)
        skill_totals = _new_totals()

        for task_idx in range(args.num_tasks):
            # Keep seeds combo-independent so combinations are directly comparable.
            task_seed = args.seed + skill_idx * 1_000_000 + task_idx
            task_rng = np.random.default_rng(task_seed)
            stats = generator.evaluate_task_attach_stats(
                task_rng,
                hard_negative_local_offsets=hard_negative_local_offsets,
            )
            _acc(skill_totals, stats)
            skill_totals["num_tasks"] += 1.0

        _acc(combo_totals, skill_totals)
        per_skill[skill] = {
            "totals": skill_totals,
            "metrics": _summarize(skill_totals),
        }

    combo_metrics = _summarize(combo_totals)
    row = {
        "attach_radius": float(attach_radius),
        "attach_capture_min_points": int(capture_min_points),
        "totals": combo_totals,
        "metrics": combo_metrics,
        "per_skill": per_skill,
    }
    return row


def _print_ranked(rows: List[dict], top_k: int) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["metrics"]["score"], reverse=True)
    top = rows_sorted[:top_k]

    print("\nAttach tuning results (ranked):")
    print(
        "rank  radius  cap_min  score    target_att  overall_att  target_miss  "
        "hardneg_fa  close_events"
    )
    for i, row in enumerate(top, start=1):
        m = row["metrics"]
        t = row["totals"]
        print(
            f"{i:>4}  {row['attach_radius']:<6.3f}  {row['attach_capture_min_points']:<7d}  "
            f"{m['score']:<7.4f}  {m['target_attach_rate']:<10.4f}  {m['attach_rate']:<11.4f}  "
            f"{m['target_miss_rate']:<11.4f}  {m['hard_negative_false_rate']:<10.4f}  {int(t['close_events'])}"
        )


def _write_csv(path: str, rows: List[dict]) -> None:
    fieldnames = [
        "attach_radius",
        "attach_capture_min_points",
        "score",
        "target_attach_rate",
        "attach_rate",
        "target_miss_rate",
        "untarget_attach_rate",
        "untarget_miss_rate",
        "hard_negative_false_rate",
        "mean_dist_attached",
        "mean_cap_attached",
        "mean_dist_failed_target",
        "mean_cap_failed_target",
        "hard_negative_probes",
        "hard_negative_false_attach",
        "close_events",
        "target_close_events",
        "untargeted_close_events",
        "attach_success",
        "attach_success_target",
        "attach_success_untargeted",
        "attach_miss_target",
        "attach_miss_untargeted",
        "num_tasks",
        "num_demos",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            m = row["metrics"]
            t = row["totals"]
            writer.writerow(
                {
                    "attach_radius": row["attach_radius"],
                    "attach_capture_min_points": row["attach_capture_min_points"],
                    "score": m["score"],
                    "target_attach_rate": m["target_attach_rate"],
                    "attach_rate": m["attach_rate"],
                    "target_miss_rate": m["target_miss_rate"],
                    "untarget_attach_rate": m["untarget_attach_rate"],
                    "untarget_miss_rate": m["untarget_miss_rate"],
                    "hard_negative_false_rate": m["hard_negative_false_rate"],
                    "mean_dist_attached": m["mean_dist_attached"],
                    "mean_cap_attached": m["mean_cap_attached"],
                    "mean_dist_failed_target": m["mean_dist_failed_target"],
                    "mean_cap_failed_target": m["mean_cap_failed_target"],
                    "hard_negative_probes": int(t["hard_negative_probes"]),
                    "hard_negative_false_attach": int(t["hard_negative_false_attach"]),
                    "close_events": int(t["close_events"]),
                    "target_close_events": int(t["target_close_events"]),
                    "untargeted_close_events": int(t["untargeted_close_events"]),
                    "attach_success": int(t["attach_success"]),
                    "attach_success_target": int(t["attach_success_target"]),
                    "attach_success_untargeted": int(t["attach_success_untargeted"]),
                    "attach_miss_target": int(t["attach_miss_target"]),
                    "attach_miss_untargeted": int(t["attach_miss_untargeted"]),
                    "num_tasks": int(t["num_tasks"]),
                    "num_demos": int(t["num_demos"]),
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Tune pseudo-demo attach gating by sweeping attach radius and capture threshold."
    )
    parser.add_argument("--shapenet_path", type=str, required=True)
    parser.add_argument("--shapenet_index_path", type=str, default=None)
    parser.add_argument("--gripper_mesh_path", type=str, required=True)

    parser.add_argument("--save_dir", type=str, default="./data/_attach_tune_tmp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=20)
    parser.add_argument("--num_demos_per_task", type=int, default=2)
    parser.add_argument(
        "--skills",
        nargs="+",
        default=DEFAULT_SKILLS,
        choices=DEFAULT_SKILLS,
        help="Waypoint skill categories used during tuning sweep.",
    )

    parser.add_argument("--num_objects_range", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--num_waypoints_range", type=int, nargs=2, default=[2, 6])
    parser.add_argument("--object_scale_range", type=float, nargs=2, default=[0.2, 0.3])
    parser.add_argument("--bias_prob", type=float, default=0.5)

    parser.add_argument("--trans_spacing", type=float, default=0.01)
    parser.add_argument("--rot_spacing_deg", type=float, default=3.0)
    parser.add_argument("--interpolation_methods", nargs="+", default=["linear", "cubic"])

    parser.add_argument("--disturbance_prob", type=float, default=0.3)
    parser.add_argument("--gripper_noise_prob", type=float, default=0.1)

    parser.add_argument("--no_attach", action="store_true")
    parser.add_argument("--attach_radius_grid", type=float, nargs="+", default=[0.015, 0.02, 0.025, 0.03])
    parser.add_argument("--attach_capture_min_points_grid", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--hard_negative_offset_m",
        type=float,
        default=0.06,
        help="Local-frame offset magnitude for hard-negative attach probes at targeted close events.",
    )

    parser.add_argument("--max_meshes", type=int, default=None)
    parser.add_argument("--cache_meshes", action="store_true")

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)

    args = parser.parse_args()

    combos = list(itertools.product(args.attach_radius_grid, args.attach_capture_min_points_grid))
    results = []

    print("Running attach tuning sweep:")
    print(f"  combos={len(combos)} skills={args.skills} num_tasks={args.num_tasks} demos_per_task={args.num_demos_per_task}")

    for radius, cap_min in combos:
        print(f"\\n[combo] attach_radius={radius:.4f}, attach_capture_min_points={cap_min}")
        row = _run_combo(args, float(radius), int(cap_min), args.skills)
        m = row["metrics"]
        print(
            "  -> "
            f"score={m['score']:.4f}, "
            f"target_attach={m['target_attach_rate']:.4f}, "
            f"overall_attach={m['attach_rate']:.4f}, "
            f"target_miss={m['target_miss_rate']:.4f}, "
            f"hardneg_false_attach={m['hard_negative_false_rate']:.4f}"
        )
        results.append(row)

    _print_ranked(results, top_k=args.top_k)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\\nSaved JSON results: {args.out_json}")

    if args.out_csv:
        _write_csv(args.out_csv, results)
        print(f"Saved CSV results: {args.out_csv}")


if __name__ == "__main__":
    main()
