import argparse
import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from ip.models.diffusion import GraphDiffusion
from ip.models.language_encoder import LanguageConditionedEncoder
from ip.utils.running_dataset import RunningDataset
from ip.configs.language_config import config as lang_config


def info_nce_loss(lang_bottleneck, demo_bottleneck, temperature):
    batch_size = lang_bottleneck.shape[0]
    lang_flat = F.normalize(lang_bottleneck.reshape(batch_size, -1), dim=-1)
    demo_flat = F.normalize(demo_bottleneck.reshape(batch_size, -1), dim=-1)

    logits = torch.matmul(lang_flat, demo_flat.t()) / temperature
    labels = torch.arange(batch_size, device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained checkpoint directory')
    parser.add_argument('--data_path_train', type=str, required=True, help='Path to language-annotated train data')
    parser.add_argument('--data_path_val', type=str, default=None, help='Optional validation data path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=100000, help='Max training steps')
    parser.add_argument('--log_every', type=int, default=200, help='Log frequency')
    parser.add_argument('--save_every', type=int, default=5000, help='Checkpoint frequency')
    parser.add_argument('--save_dir', type=str, default='./runs_lang', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--use_wandb', type=int, default=0, help='Enable Weights & Biases logging [0,1]')
    parser.add_argument('--wandb_project', type=str, default='Instant Policy', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (optional)')
    parser.add_argument('--wandb_run_name', type=str, default='lang_train', help='W&B run name')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    base_config = pickle.load(open(f'{args.model_path}/config.pkl', 'rb'))
    base_config['device'] = args.device
    base_config['batch_size'] = args.batch_size

    # Merge language config values.
    model_config = dict(base_config)
    model_config.update(lang_config)
    model_config['device'] = args.device
    model_config['batch_size'] = args.batch_size

    teacher = GraphDiffusion.load_from_checkpoint(
        f'{args.model_path}/model.pt',
        config=base_config,
        strict=True,
        map_location=args.device
    ).to(args.device)
    teacher.eval()

    freeze_module(teacher)

    lang_encoder = LanguageConditionedEncoder(model_config).to(args.device)
    optimizer = torch.optim.AdamW(lang_encoder.parameters(),
                                  lr=model_config['lang_lr'],
                                  weight_decay=model_config['lang_weight_decay'])

    use_wandb = bool(args.use_wandb)
    if use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError('wandb is required when --use_wandb=1') from exc
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   name=args.wandb_run_name,
                   config={
                       'batch_size': args.batch_size,
                       'max_steps': args.max_steps,
                       'lang_lr': model_config['lang_lr'],
                       'lang_weight_decay': model_config['lang_weight_decay'],
                       'contrastive_temperature': model_config['contrastive_temperature'],
                       'contrastive_weight': model_config['contrastive_weight'],
                       'l2_weight': model_config['l2_weight'],
                   })

    dset = RunningDataset(args.data_path_train, len(os.listdir(args.data_path_train)),
                          rand_g_prob=0.0, require_lang=True)
    dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    if args.data_path_val:
        dset_val = RunningDataset(args.data_path_val, len(os.listdir(args.data_path_val)),
                                  rand_g_prob=0.0, require_lang=True)
        dataloader_val = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        dataloader_val = None

    global_step = 0
    for epoch in range(1000000):
        for data in dataloader:
            if global_step >= args.max_steps:
                break

            data = data.to(args.device)
            batch_size = data.actions.shape[0]

            with torch.no_grad():
                demo_bottleneck = teacher.model.get_demo_bottleneck(data)

            with torch.no_grad():
                teacher.model._ensure_scene_embeddings(data)
                teacher.model._populate_action_scene_embeddings(data)
                teacher.model.graph.update_graph(data)
                x_dict = teacher.model.local_encoder(
                    teacher.model.graph.graph.x_dict,
                    teacher.model.graph.graph.edge_index_dict,
                    teacher.model.graph.graph.edge_attr_dict)

                g_mask = teacher.model.graph.graph.gripper_time == teacher.model.traj_horizon
                s_mask = teacher.model.graph.graph.scene_traj == teacher.model.traj_horizon

                current_gripper_x = x_dict['gripper'][g_mask].view(batch_size,
                                                                   teacher.model.graph.num_g_nodes, -1)
                current_gripper_pos = teacher.model.graph.graph['gripper'].pos[g_mask].view(
                    batch_size, teacher.model.graph.num_g_nodes, 3)
                current_scene_x = x_dict['scene'][s_mask].view(batch_size,
                                                               teacher.model.num_scenes_nodes, -1)
                current_scene_pos = teacher.model.graph.graph['scene'].pos[s_mask].view(
                    batch_size, teacher.model.num_scenes_nodes, 3)

            lang_bottleneck = lang_encoder(current_scene_x, current_scene_pos,
                                           current_gripper_x, current_gripper_pos,
                                           data.lang_emb.view(batch_size, -1))

            contrastive = info_nce_loss(lang_bottleneck, demo_bottleneck,
                                        model_config['contrastive_temperature'])
            l2_loss = F.mse_loss(lang_bottleneck, demo_bottleneck)
            loss = (model_config['contrastive_weight'] * contrastive +
                    model_config['l2_weight'] * l2_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    sim = F.cosine_similarity(
                        lang_bottleneck.reshape(batch_size, -1),
                        demo_bottleneck.reshape(batch_size, -1),
                        dim=-1
                    ).mean().item()
                print(f'step {global_step} loss {loss.item():.4f} '
                      f'contrastive {contrastive.item():.4f} l2 {l2_loss.item():.4f} sim {sim:.3f}')
                if use_wandb:
                    wandb.log({
                        'loss': loss.item(),
                        'contrastive': contrastive.item(),
                        'l2_loss': l2_loss.item(),
                        'sim': sim,
                    }, step=global_step)

            if global_step % args.save_every == 0 and global_step > 0:
                ckpt_path = os.path.join(args.save_dir, f'lang_encoder_{global_step}.pt')
                torch.save(lang_encoder.state_dict(), ckpt_path)

            global_step += 1

        if global_step >= args.max_steps:
            break

        if dataloader_val is not None:
            lang_encoder.eval()
            sims = []
            with torch.no_grad():
                for data in dataloader_val:
                    data = data.to(args.device)
                    batch_size = data.actions.shape[0]
                    demo_bottleneck = teacher.model.get_demo_bottleneck(data)

                    teacher.model._ensure_scene_embeddings(data)
                    teacher.model._populate_action_scene_embeddings(data)
                    teacher.model.graph.update_graph(data)
                    x_dict = teacher.model.local_encoder(
                        teacher.model.graph.graph.x_dict,
                        teacher.model.graph.graph.edge_index_dict,
                        teacher.model.graph.graph.edge_attr_dict)

                    g_mask = teacher.model.graph.graph.gripper_time == teacher.model.traj_horizon
                    s_mask = teacher.model.graph.graph.scene_traj == teacher.model.traj_horizon

                    current_gripper_x = x_dict['gripper'][g_mask].view(batch_size,
                                                                       teacher.model.graph.num_g_nodes, -1)
                    current_gripper_pos = teacher.model.graph.graph['gripper'].pos[g_mask].view(
                        batch_size, teacher.model.graph.num_g_nodes, 3)
                    current_scene_x = x_dict['scene'][s_mask].view(batch_size,
                                                                   teacher.model.num_scenes_nodes, -1)
                    current_scene_pos = teacher.model.graph.graph['scene'].pos[s_mask].view(
                        batch_size, teacher.model.num_scenes_nodes, 3)

                    lang_bottleneck = lang_encoder(current_scene_x, current_scene_pos,
                                                   current_gripper_x, current_gripper_pos,
                                                   data.lang_emb.view(batch_size, -1))
                    sim = F.cosine_similarity(
                        lang_bottleneck.reshape(batch_size, -1),
                        demo_bottleneck.reshape(batch_size, -1),
                        dim=-1
                    )
                    sims.append(sim.mean().item())

            print(f'val sim {sum(sims) / len(sims):.3f}')
            if use_wandb:
                wandb.log({'val_sim': sum(sims) / len(sims)}, step=global_step)
            lang_encoder.train()

    ckpt_path = os.path.join(args.save_dir, 'lang_encoder_last.pt')
    torch.save(lang_encoder.state_dict(), ckpt_path)
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
