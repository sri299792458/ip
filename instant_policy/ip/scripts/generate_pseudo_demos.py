import argparse
import os

import numpy as np
import torch

from ip.generation.config import GenerationConfig
from ip.generation.pseudo_demo_generator import PseudoDemoGenerator
from ip.models.scene_encoder import SceneEncoder


def build_config(args):
    config = GenerationConfig(
        shapenet_path=args.shapenet_path,
        shapenet_index_path=args.shapenet_index_path,
        save_dir=args.save_dir,
        num_tasks=args.num_tasks,
        num_demos_per_task=tuple(args.num_demos_per_task),
        num_context_demos=args.num_context_demos,
        randomize_num_demos=args.randomize_num_demos,
        num_context_range=tuple(args.num_context_range),
        num_waypoints_range=tuple(args.num_waypoints_range),
        bias_prob=args.bias_prob,
        num_objects_range=tuple(args.num_objects_range),
        object_scale_range=tuple(args.object_scale_range),
        trans_spacing=args.trans_spacing,
        rot_spacing_deg=args.rot_spacing_deg,
        disturbance_prob=args.disturbance_prob,
        gripper_noise_prob=args.gripper_noise_prob,
        attach_on_grasp=not args.no_attach,
        seed=args.seed,
        save_renders=args.save_renders,
        render_dir=args.render_dir,
        render_stride=args.render_stride,
        render_visual_camera=args.render_visual_camera,
        render_save_depth=args.render_save_depth,
        render_make_videos=args.render_make_videos,
        render_video_dir=args.render_video_dir,
        render_video_fps=args.render_video_fps,
        render_video_ext=args.render_video_ext,
    )
    return config


def load_scene_encoder(args):
    if not args.compute_embeddings:
        return None
    scene_encoder = SceneEncoder(num_freqs=10, embd_dim=512)
    scene_encoder.load_state_dict(torch.load(args.scene_encoder_path, map_location=args.device))
    scene_encoder = scene_encoder.to(args.device)
    scene_encoder.eval()
    return scene_encoder


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo demonstrations for Instant Policy.")
    parser.add_argument("--shapenet_path", type=str, required=True)
    parser.add_argument("--shapenet_index_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_tasks", type=int, default=1000)
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--fill_buffer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num_objects_range", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--num_waypoints_range", type=int, nargs=2, default=[2, 6])
    parser.add_argument("--num_demos_per_task", type=int, nargs=2, default=[3, 5])
    parser.add_argument("--num_context_demos", type=int, default=2)
    parser.add_argument("--num_context_range", type=int, nargs=2, default=[1, 5])
    parser.add_argument("--randomize_num_demos", action="store_true")
    parser.add_argument("--bias_prob", type=float, default=0.5)
    parser.add_argument("--object_scale_range", type=float, nargs=2, default=[0.05, 0.15])
    parser.add_argument("--trans_spacing", type=float, default=0.01)
    parser.add_argument("--rot_spacing_deg", type=float, default=3.0)
    parser.add_argument("--disturbance_prob", type=float, default=0.3)
    parser.add_argument("--gripper_noise_prob", type=float, default=0.1)
    parser.add_argument("--no_attach", action="store_true")
    parser.add_argument("--save_renders", action="store_true")
    parser.add_argument("--render_dir", type=str, default=None)
    parser.add_argument("--render_stride", type=int, default=1)
    parser.add_argument("--render_visual_camera", type=int, default=0)
    parser.add_argument("--render_save_depth", action="store_true")
    parser.add_argument("--render_make_videos", action="store_true")
    parser.add_argument("--render_video_dir", type=str, default=None)
    parser.add_argument("--render_video_fps", type=int, default=15)
    parser.add_argument("--render_video_ext", type=str, default="mp4")

    parser.add_argument("--compute_embeddings", action="store_true")
    parser.add_argument("--scene_encoder_path", type=str, default="./checkpoints/scene_encoder.pt")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)

    config = build_config(args)
    scene_encoder = load_scene_encoder(args)
    generator = PseudoDemoGenerator(config, scene_encoder=scene_encoder)
    generator.generate_dataset(
        num_tasks=args.num_tasks,
        save_dir=args.save_dir,
        task_start=args.task_start,
        append=args.append,
        buffer_size=args.buffer_size,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        fill_buffer=args.fill_buffer,
    )


if __name__ == "__main__":
    main()
