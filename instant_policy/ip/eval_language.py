import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from ip.models.diffusion import GraphDiffusion
from ip.models.language_encoder import LanguageConditionedEncoder
from ip.utils.common_utils import pose_to_transform, transform_to_pose
from ip.utils.common_utils import actions_to_transforms, transforms_to_actions, get_rigid_transforms
from ip.utils.data_proc import save_sample, subsample_pcd, transform_pcd
from ip.utils.rl_bench_utils import get_point_cloud
from ip.utils.rl_bench_tasks import TASK_NAMES
from ip.utils.language_utils import get_language_description, encode_texts


def test_step_with_bottleneck(model, data, bottleneck):
    batch_size = data.actions.shape[0]
    noisy_actions = torch.randn(
        (batch_size, model.config['pre_horizon'], 6), device=model.device
    )
    noisy_actions = torch.clamp(noisy_actions, -1, 1)
    noisy_actions = model.normalizer.denormalize_actions(noisy_actions)
    noisy_actions = actions_to_transforms(noisy_actions.view(-1, 6)).view(batch_size, -1, 4, 4)

    noisy_grips = torch.randn((batch_size, model.config['pre_horizon'], 1), device=model.device)
    noisy_grips = torch.clamp(noisy_grips, -1, 1)

    model.noise_scheduler.set_timesteps(model.config['num_diffusion_iters_test'])

    for k in range(model.config['num_diffusion_iters_test'] - 1, -1, -1):
        data.actions = noisy_actions
        data.actions_grip = noisy_grips.squeeze(-1)
        data.diff_time = torch.tensor([[
            k if k != model.config['num_diffusion_iters_test'] - 1 else model.config['num_diffusion_iters_train']
        ]] * batch_size, device=model.device)

        preds = model.model.forward_from_bottleneck(data, bottleneck)
        preds[..., :6] = model.normalizer.denormalize_labels(preds[..., :6])

        current_gripper_pos = model.model.get_transformed_node_pos(noisy_actions, transform=False)
        mode_output = preds[..., 3:6] + current_gripper_pos + torch.mean(preds[..., :3], dim=-2, keepdim=True)

        pred_gripper_pos = model.noise_scheduler.step(
            model_output=mode_output,
            sample=current_gripper_pos,
            timestep=k,
        ).prev_sample

        T_e_e = get_rigid_transforms(
            current_gripper_pos.view(-1, pred_gripper_pos.shape[-2], 3),
            pred_gripper_pos.view(-1, pred_gripper_pos.shape[-2], 3)
        ).view(batch_size, -1, 4, 4)

        noisy_actions = torch.matmul(noisy_actions, T_e_e)

        noisy_grips = model.noise_scheduler.step(
            model_output=preds[..., -1:].mean(dim=-2),
            sample=noisy_grips,
            timestep=k,
        ).prev_sample
        noisy_grips = torch.clamp(noisy_grips, -1, 1)

        noisy_actions_6d = transforms_to_actions(noisy_actions.view(-1, 4, 4)).view(batch_size, -1, 6)
        noisy_actions_6d = model.normalizer.normalize_actions(noisy_actions_6d)
        noisy_actions_6d = torch.clamp(noisy_actions_6d, -1, 1)
        noisy_actions_6d = model.normalizer.denormalize_actions(noisy_actions_6d)
        noisy_actions = actions_to_transforms(noisy_actions_6d.view(-1, 6)).view(batch_size, -1, 4, 4)

    return noisy_actions, torch.sign(noisy_grips)


def compute_language_bottleneck(model, lang_encoder, data, lang_emb):
    agi = model.model
    agi._ensure_scene_embeddings(data)
    agi._populate_action_scene_embeddings(data)
    agi._ensure_diff_time(data)
    agi.graph.update_graph(data)

    with torch.no_grad():
        x_dict = agi.local_encoder(
            agi.graph.graph.x_dict,
            agi.graph.graph.edge_index_dict,
            agi.graph.graph.edge_attr_dict
        )

    g_mask = agi.graph.graph.gripper_time == agi.traj_horizon
    s_mask = agi.graph.graph.scene_traj == agi.traj_horizon

    current_gripper_x = x_dict['gripper'][g_mask].view(1, agi.graph.num_g_nodes, -1)
    current_gripper_pos = agi.graph.graph['gripper'].pos[g_mask].view(1, agi.graph.num_g_nodes, 3)
    current_scene_x = x_dict['scene'][s_mask].view(1, agi.num_scenes_nodes, -1)
    current_scene_pos = agi.graph.graph['scene'].pos[s_mask].view(1, agi.num_scenes_nodes, 3)

    with torch.no_grad():
        return lang_encoder(current_scene_x, current_scene_pos,
                            current_gripper_x, current_gripper_pos,
                            lang_emb)


def _configure_env(env, task, restrict_rot):
    def temp(position, euler=None, quaternion=None, ignore_collisions=False, trials=300, max_configs=1,
             distance_threshold=0.65, max_time_ms=10, trials_per_goal=1, algorithm=None, relative_to=None):
        return env._robot.arm.get_linear_path(position, euler, quaternion, ignore_collisions=ignore_collisions,
                                              relative_to=relative_to)

    env._robot.arm.get_path = temp
    env._scene._start_arm_joint_pos = np.array([6.74760377e-05, -1.91104114e-02, -3.62065766e-05, -1.64271665e+00,
                                                -1.14094291e-07, 1.55336857e+00, 7.85427451e-01])

    rot_bounds = env._scene.task.base_rotation_bounds()
    mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
    if restrict_rot:
        env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
                                                        (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3)))


def _init_env(task_name, headless, restrict_rot):
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode,
                      './',
                      obs_config=obs_config,
                      headless=headless)
    env.launch()
    if not headless:
        if hasattr(env._pyrep, "set_rendering"):
            env._pyrep.set_rendering(True)
        if hasattr(env._pyrep, "step_ui"):
            env._pyrep.step_ui()

    task = env.get_task(TASK_NAMES[task_name])
    _configure_env(env, task, restrict_rot)
    return env, task



def rollout_model_language(model, lang_encoder, lang_emb, task_name='phone_on_base', max_execution_steps=30,
                           execution_horizon=8, num_rollouts=2, headless=False, num_traj_wp=10, restrict_rot=True,
                           env=None, task=None, shutdown_env=True):
    created_env = False
    if env is None or task is None:
        env, task = _init_env(task_name, headless, restrict_rot)
        created_env = True
    else:
        _configure_env(env, task, restrict_rot)

    # Dummy demos to satisfy data formatting (not used in language path).
    dummy_demo = None
    num_demos = model.model.num_demos
    successes = []
    pbar = trange(num_rollouts, desc=f'Evaluating model, SR: 0/{num_rollouts}', leave=False)
    for i in pbar:
        done = False
        while not done:
            try:
                task.reset()
                done = True
            except:
                continue

        env_action = np.zeros(8)
        success = 0
        for k in range(max_execution_steps):
            curr_obs = task.get_observation()
            T_w_e = pose_to_transform(curr_obs.gripper_pose)
            current_pcd = transform_pcd(subsample_pcd(get_point_cloud(curr_obs)),
                                        np.linalg.inv(T_w_e))

            if dummy_demo is None:
                num_points = current_pcd.shape[0]
                demo_obs = [np.zeros((num_points, 3), dtype=np.float32) for _ in range(num_traj_wp)]
                demo_T = [np.eye(4, dtype=np.float32) for _ in range(num_traj_wp)]
                demo_grip = [0.0 for _ in range(num_traj_wp)]
                dummy_demo = {'obs': demo_obs, 'grips': demo_grip, 'T_w_es': demo_T}

            full_sample = {
                'demos': [dummy_demo for _ in range(num_demos)],
                'live': {
                    'obs': [current_pcd],
                    'grips': [curr_obs.gripper_open],
                    'actions_grip': [np.zeros(8)],
                    'T_w_es': [T_w_e],
                    'actions': [T_w_e.reshape(1, 4, 4).repeat(8, axis=0)],
                }
            }

            data = save_sample(full_sample, None)

            if k == 0:
                demo_scene_node_embds, demo_scene_node_pos = model.model.get_demo_scene_emb(
                    data.to(model.config['device']))
            live_scene_node_embds, live_scene_node_pos = model.model.get_live_scene_emb(data.to(model.config['device']))
            data.live_scene_node_embds = live_scene_node_embds.clone()
            data.live_scene_node_pos = live_scene_node_pos.clone()
            data.demo_scene_node_embds = demo_scene_node_embds.clone()
            data.demo_scene_node_pos = demo_scene_node_pos.clone()

            lang_bottleneck = compute_language_bottleneck(model, lang_encoder, data.to(model.config['device']), lang_emb)

            with torch.no_grad():
                with torch.autocast(dtype=torch.float32, device_type=model.config['device']):
                    actions, grips = test_step_with_bottleneck(model, data.to(model.config['device']), lang_bottleneck)
                actions = actions.squeeze().cpu().numpy()
                grips = grips.squeeze().cpu().numpy()

            for j in range(execution_horizon):
                env_action[:7] = transform_to_pose(T_w_e @ actions[j])
                env_action[7] = int((grips[j] + 1) / 2 > 0.5)
                try:
                    curr_obs, reward, terminate = task.step(env_action)
                    success = int(terminate and reward > 0.)
                except Exception:
                    terminate = True
                if terminate:
                    break

            else:
                continue
            break
        successes.append(success)
        pbar.set_description(f'Evaluating model, SR: {sum(successes)}/{len(successes)}')
        pbar.refresh()
    pbar.close()
    if created_env and shutdown_env:
        env.shutdown()
    return sum(successes) / len(successes)


def _load_paraphrases(path):
    with open(path, 'r', encoding='utf-8') as handle:
        lines = []
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            lines.append(stripped)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='plate_out')
    parser.add_argument('--num_rollouts', type=int, default=5)
    parser.add_argument('--restrict_rot', type=int, default=1)
    parser.add_argument('--compile_models', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='./checkpoints')
    parser.add_argument('--lang_encoder_path', type=str, required=True)
    parser.add_argument('--lang_text', type=str, default=None)
    parser.add_argument('--lang_emb_path', type=str, default=None)
    parser.add_argument('--paraphrase_file', type=str, default=None)
    parser.add_argument('--lang_model_name', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    restrict_rot = bool(args.restrict_rot)
    compile_models = bool(args.compile_models)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    config = pickle.load(open(f'{args.model_path}/config.pkl', 'rb'))
    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_diffusion_iters_test'] = 8
    config['device'] = args.device

    model = GraphDiffusion.load_from_checkpoint(f'{args.model_path}/model.pt', config=config, strict=True,
                                                map_location=config['device']).to(config['device'])
    model.model.reinit_graphs(1, num_demos=config['num_demos'])
    model.eval()

    if compile_models:
        model.model.compile_models()

    lang_config = dict(config)
    lang_config['lang_emb_dim'] = 768
    lang_config['lang_num_layers'] = 4

    lang_encoder = LanguageConditionedEncoder(lang_config).to(config['device'])
    lang_encoder.load_state_dict(torch.load(args.lang_encoder_path, map_location=config['device']))
    lang_encoder.eval()

    if args.paraphrase_file:
        if args.lang_emb_path:
            raise ValueError('--lang_emb_path is not supported with --paraphrase_file')
        paraphrases = _load_paraphrases(args.paraphrase_file)
        if not paraphrases:
            raise ValueError(f'No paraphrases found in {args.paraphrase_file}')

        base_text = args.lang_text or get_language_description(args.task_name)
        if base_text not in paraphrases:
            texts = [base_text] + paraphrases
            base_idx = 0
        else:
            texts = paraphrases
            base_idx = texts.index(base_text)

        embeddings = encode_texts(texts, model_name=args.lang_model_name, device=args.device)
        base_emb = embeddings[base_idx]

        env, task = _init_env(args.task_name, headless=False, restrict_rot=restrict_rot)
        results = []
        for text, emb in zip(texts, embeddings):
            sim = F.cosine_similarity(emb.unsqueeze(0), base_emb.unsqueeze(0), dim=-1).item()
            sr = rollout_model_language(model, lang_encoder, emb.unsqueeze(0), args.task_name,
                                        num_rollouts=args.num_rollouts, execution_horizon=8,
                                        num_traj_wp=config['traj_horizon'], restrict_rot=restrict_rot,
                                        env=env, task=task, shutdown_env=False)
            results.append((text, sr, sim))
            print(f'sr={sr:.3f} sim={sim:.3f} text="{text}"')
        env.shutdown()

        srs = np.array([sr for _, sr, _ in results], dtype=np.float32)
        print(f'Paraphrase SR mean={srs.mean():.3f} std={srs.std():.3f}')
    else:
        if args.lang_emb_path:
            lang_emb = torch.load(args.lang_emb_path, map_location=config['device']).unsqueeze(0)
            lang_emb = lang_emb.to(config['device'])
        else:
            if args.lang_text:
                lang_text = args.lang_text
            else:
                lang_text = get_language_description(args.task_name)
            lang_emb = encode_texts([lang_text], model_name=args.lang_model_name, device=args.device)

        sr = rollout_model_language(model, lang_encoder, lang_emb, args.task_name,
                                    num_rollouts=args.num_rollouts, execution_horizon=8,
                                    num_traj_wp=config['traj_horizon'], restrict_rot=restrict_rot)
        print('Success rate:', sr)


if __name__ == '__main__':
    main()
