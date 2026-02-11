import argparse
import os

from ip.generation_bimanual import (
    BimanualGenerationConfig,
    BimanualPseudoDemoGenerator,
    PERACT2_BIMANUAL_TASKS,
)


def parse_weights(s: str):
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        return None
    return vals


def main():
    p = argparse.ArgumentParser(
        description="Generate bimanual pseudo demos with ShapeNet scenes + Robotiq gripper mesh."
    )

    p.add_argument("--shapenet_path", type=str, required=True)
    p.add_argument("--gripper_mesh_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--shapenet_index_path", type=str, default=None)
    p.add_argument("--max_meshes", type=int, default=None)
    p.add_argument("--cache_meshes", action="store_true")
    p.add_argument("--surface_sample_count", type=int, default=512)

    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--pred_horizon", type=int, default=8)
    p.add_argument("--min_steps", type=int, default=14)
    p.add_argument("--max_steps", type=int, default=24)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--pcd_storage_dtype", type=str, default="float32", choices=["float32", "float16"])

    p.add_argument("--num_objects_range", type=int, nargs=2, default=[3, 5])
    p.add_argument("--object_scale_range", type=float, nargs=2, default=[0.2, 0.3])

    p.add_argument("--attach_capture_min_points", type=int, default=3)

    p.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Subset of tasks to sample. Defaults to the 13 RLBench2 bimanual tasks.",
    )
    p.add_argument(
        "--forced_task",
        type=str,
        default=None,
        choices=PERACT2_BIMANUAL_TASKS,
        help="Force one task type for debugging.",
    )
    p.add_argument(
        "--task_weights",
        type=str,
        default=None,
        help="Comma-separated weights matching --tasks order. Example: '1,1,2,2'.",
    )

    p.add_argument("--save_renders", action="store_true")
    p.add_argument("--render_dir", type=str, default=None)
    p.add_argument("--render_stride", type=int, default=1)
    p.add_argument("--render_visual_camera", type=int, default=0)
    p.add_argument("--render_visual_width", type=int, default=640)
    p.add_argument("--render_visual_height", type=int, default=640)
    p.add_argument("--render_save_depth", action="store_true")
    p.add_argument("--render_make_videos", action="store_true")
    p.add_argument("--render_video_dir", type=str, default=None)
    p.add_argument("--render_video_fps", type=int, default=15)
    p.add_argument("--render_video_ext", type=str, default="mp4")

    p.add_argument("--task_start", type=int, default=0)
    p.add_argument("--append", action="store_true")
    p.add_argument("--buffer_size", type=int, default=None)
    p.add_argument("--fill_buffer", action="store_true")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)

    args = p.parse_args()

    task_names = args.tasks if args.tasks else list(PERACT2_BIMANUAL_TASKS)
    for t in task_names:
        if t not in PERACT2_BIMANUAL_TASKS:
            raise ValueError(f"Unknown task '{t}'. Expected one of {PERACT2_BIMANUAL_TASKS}")

    task_weights = None
    if args.task_weights is not None:
        task_weights = parse_weights(args.task_weights)

    cfg = BimanualGenerationConfig(
        shapenet_path=args.shapenet_path,
        shapenet_index_path=args.shapenet_index_path,
        gripper_mesh_path=args.gripper_mesh_path,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        pred_horizon=args.pred_horizon,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        num_points=args.num_points,
        pcd_storage_dtype=args.pcd_storage_dtype,
        num_objects_range=(int(args.num_objects_range[0]), int(args.num_objects_range[1])),
        object_scale_range=(float(args.object_scale_range[0]), float(args.object_scale_range[1])),
        max_meshes=args.max_meshes,
        cache_meshes=bool(args.cache_meshes),
        surface_sample_count=int(args.surface_sample_count),
        attach_capture_min_points=int(args.attach_capture_min_points),
        task_names=task_names,
        task_weights=task_weights,
        forced_task=args.forced_task,
        save_renders=bool(args.save_renders),
        render_dir=args.render_dir,
        render_stride=int(args.render_stride),
        render_visual_camera=int(args.render_visual_camera),
        render_visual_width=int(args.render_visual_width),
        render_visual_height=int(args.render_visual_height),
        render_save_depth=bool(args.render_save_depth),
        render_make_videos=bool(args.render_make_videos),
        render_video_dir=args.render_video_dir,
        render_video_fps=int(args.render_video_fps),
        render_video_ext=args.render_video_ext,
        task_start=args.task_start,
        append=args.append,
        buffer_size=args.buffer_size,
        fill_buffer=args.fill_buffer,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    generator = BimanualPseudoDemoGenerator(cfg)
    generator.generate_dataset()


if __name__ == "__main__":
    main()
