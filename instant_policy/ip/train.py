from ip.models.diffusion import *
from ip.configs.base_config import config
import pickle
import os
import torch
import tempfile
from ip.utils.running_dataset import RunningDataset
from ip.utils.trajectory_dataset import TrajectoryDataset
from torch_geometric.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import argparse
from glob import glob


def _latest_resume_checkpoint(save_dir: str):
    if not os.path.isdir(save_dir):
        return None
    last_ckpt = os.path.join(save_dir, "last.pt")
    if os.path.isfile(last_ckpt):
        return last_ckpt

    numbered = []
    for path in glob(os.path.join(save_dir, "*.pt")):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.isdigit():
            numbered.append((int(stem), path))
    if numbered:
        numbered.sort(key=lambda x: x[0])
        return numbered[-1][1]

    best_ckpt = os.path.join(save_dir, "best.pt")
    if os.path.isfile(best_ckpt):
        return best_ckpt
    return None


def _norm_orig_mod_key(key: str) -> str:
    return key.replace("._orig_mod.", ".")


def _align_checkpoint_state_dict_for_model(ckpt_path: str, model: torch.nn.Module) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt or not isinstance(ckpt["state_dict"], dict):
        return ckpt_path

    src_sd = ckpt["state_dict"]
    src_keys = list(src_sd.keys())
    tgt_keys = list(model.state_dict().keys())
    if set(src_keys) == set(tgt_keys):
        return ckpt_path

    src_norm = {_norm_orig_mod_key(k): k for k in src_keys}
    tgt_norm = {_norm_orig_mod_key(k): k for k in tgt_keys}
    if len(src_norm) != len(src_keys) or len(tgt_norm) != len(tgt_keys):
        return ckpt_path
    if set(src_norm.keys()) != set(tgt_norm.keys()):
        return ckpt_path

    aligned_sd = {}
    for tgt_key in tgt_keys:
        nkey = _norm_orig_mod_key(tgt_key)
        aligned_sd[tgt_key] = src_sd[src_norm[nkey]]

    ckpt["state_dict"] = aligned_sd

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    fd, aligned_path = tempfile.mkstemp(
        prefix=".resume_aligned_", suffix=".pt", dir=ckpt_dir
    )
    os.close(fd)
    torch.save(ckpt, aligned_path)
    print(
        f"Adjusted checkpoint state_dict keys for current model format:\n"
        f"  source={ckpt_path}\n  aligned={aligned_path}"
    )
    return aligned_path


def _debug_dataset_path(label: str, path: str, pattern: str, max_items: int = 20):
    abs_path = os.path.abspath(path)
    matches = sorted(glob(os.path.join(path, pattern)))
    print(f"[PATH_DEBUG] {label} cwd={os.getcwd()}")
    print(f"[PATH_DEBUG] {label} path={path} abs={abs_path} exists={os.path.isdir(path)}")
    print(f"[PATH_DEBUG] {label} pattern={pattern} count={len(matches)} sample={matches[:max_items]}")

    parent = os.path.dirname(path.rstrip(os.sep)) or os.sep
    try:
        parent_entries = sorted(os.listdir(parent))[:50]
    except Exception as exc:
        parent_entries = [f"<list-error: {exc}>"]
    print(f"[PATH_DEBUG] {label} parent={parent} entries={parent_entries}")


if __name__ == '__main__':
    # Prefer Tensor Core throughput on modern GPUs (A100, etc.).
    torch.set_float32_matmul_precision('high')
    ####################################################################################################################
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='test', help='Name of the run.')
    parser.add_argument('--record', type=int, default=0,
                        help='Whether to log the training and save models [0, 1].')
    parser.add_argument('--use_wandb', type=int, default=1,
                        help='Log training on weights and biases [0, 1]. You might need to log in to wandb.')
    parser.add_argument('--save_path', type=str, default='./runs',
                        help='Where the config and models will be saved.')
    parser.add_argument('--fine_tune', type=int, default=0,
                        help='Whether to train from scratch (0), or fine-tune existing model (1).')
    parser.add_argument('--model_path', type=str, default='./checkpoints',
                        help='If fine-tuning, path to where that model is saved.')
    parser.add_argument('--model_name', type=str, default='model.pt',
                        help='If fine-tuning, path to what is the name of the model.')
    parser.add_argument('--compile_models', type=int, default=0,
                        help='Whether to torch.compile model modules [0, 1].')
    parser.add_argument('--data_path_train', type=str, default='./data/train',
                        help='Path to the training data.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for fine-tuning. When not fine-tuning, it is defined in the config')
    parser.add_argument('--data_path_val', type=str, default='./data/val',
                        help='Path to the validation data.')
    parser.add_argument('--data_format', type=str, default='steps', choices=['steps', 'trajectory'],
                        help='Dataset format: steps (data_*.pt) or trajectory (task_*.pt).')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Points per point cloud (trajectory format only).')
    parser.add_argument('--subsample_live', action='store_true',
                        help='Subsample live trajectories before sampling a step (trajectory format only).')
    parser.add_argument('--live_spacing_trans', type=float, default=0.01,
                        help='Translation spacing for subsample_live (trajectory format only).')
    parser.add_argument('--live_spacing_rot', type=float, default=3.0,
                        help='Rotation spacing (degrees) for subsample_live (trajectory format only).')
    parser.add_argument('--num_iters_override', type=int, default=None,
                        help='Optional override for total training steps (useful for throughput benchmarking).')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader worker count.')
    parser.add_argument('--persistent_workers', type=int, default=1,
                        help='Use persistent DataLoader workers when num_workers > 0 [0, 1].')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='DataLoader prefetch_factor when num_workers > 0.')
    parser.add_argument('--val_check_interval', type=int, default=20000,
                        help='Validation check interval in optimizer steps.')
    parser.add_argument('--log_every_n_steps', type=int, default=500,
                        help='Logging interval in optimizer steps.')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs/devices for Lightning trainer.')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='Lightning strategy, e.g. auto or ddp.')
    parser.add_argument('--resume_ckpt_path', type=str, default=None,
                        help='Resume full trainer state from this checkpoint path.')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Auto-resume from latest checkpoint in <save_path>/<run_name>.')
    parser.add_argument('--wandb_id', type=str, default=None,
                        help='Optional W&B run id to resume the same online run.')
    parser.add_argument('--wandb_resume', type=str, default='allow', choices=['allow', 'must', 'never'],
                        help='W&B resume policy when wandb logging is enabled.')

    args = parser.parse_args()
    debug_paths = os.environ.get("IP_DEBUG_PATHS", "0") == "1"

    record = bool(args.record)
    use_wandb = bool(args.use_wandb)
    fine_tune = bool(args.fine_tune)
    compile_models = bool(args.compile_models)
    run_name = args.run_name
    save_path = args.save_path
    model_path = args.model_path
    model_name = args.model_name
    data_path_train = args.data_path_train
    data_path_val = args.data_path_val
    data_format = args.data_format
    num_points = args.num_points
    subsample_live = args.subsample_live
    live_spacing_trans = args.live_spacing_trans
    live_spacing_rot = args.live_spacing_rot
    num_iters_override = args.num_iters_override
    num_workers = int(args.num_workers)
    persistent_workers = bool(args.persistent_workers)
    prefetch_factor = int(args.prefetch_factor)
    val_check_interval = int(args.val_check_interval)
    log_every_n_steps = int(args.log_every_n_steps)
    trainer_devices = int(args.devices)
    trainer_strategy = args.strategy
    bs = args.batch_size
    ####################################################################################################################
    save_dir = f'{save_path}/{run_name}' if record else None
    resume_search_dir = os.path.join(save_path, run_name)
    resume_ckpt_path = args.resume_ckpt_path

    if record and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.auto_resume and resume_ckpt_path is None:
        resume_ckpt_path = _latest_resume_checkpoint(resume_search_dir)
        if resume_ckpt_path is None:
            raise RuntimeError(
                f"--auto_resume requested, but no checkpoint found in {resume_search_dir}"
            )

    if resume_ckpt_path is not None and not os.path.isfile(resume_ckpt_path):
        raise RuntimeError(f"Resume checkpoint not found: {resume_ckpt_path}")

    if resume_ckpt_path is not None:
        # For true resume, use the run-local config when available.
        resume_cfg_path = os.path.join(os.path.dirname(resume_ckpt_path), 'config.pkl')
        if os.path.isfile(resume_cfg_path):
            cfg = pickle.load(open(resume_cfg_path, 'rb'))
        elif fine_tune:
            cfg = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
        else:
            cfg = dict(config)
        cfg['compile_models'] = compile_models
        cfg['batch_size'] = bs
        cfg['save_dir'] = save_dir
        cfg['record'] = record
        model = GraphDiffusion(cfg).to(cfg['device'])
    else:
        if fine_tune:
            cfg = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
            cfg['compile_models'] = compile_models
            cfg['batch_size'] = bs
            cfg['save_dir'] = save_dir
            cfg['record'] = record
            # Warm-start from checkpoint weights (not trainer-state resume).
            model = GraphDiffusion.load_from_checkpoint(
                f'{model_path}/{model_name}',
                config=cfg,
                strict=True,
                map_location=cfg['device'],
            ).to(cfg['device'])
        else:
            cfg = dict(config)
            cfg['compile_models'] = compile_models
            cfg['save_dir'] = save_dir
            cfg['record'] = record
            model = GraphDiffusion(cfg).to(cfg['device'])

    if num_iters_override is not None:
        if int(num_iters_override) < 1:
            raise ValueError('--num_iters_override must be >= 1')
        cfg['num_iters'] = int(num_iters_override)
    ####################################################################################################################
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    if data_format == 'trajectory':
        val_files = sorted(glob(os.path.join(data_path_val, 'task_*.pt')))
        train_files = sorted(glob(os.path.join(data_path_train, 'task_*.pt')))
        val_count = len(val_files)
        train_count = len(train_files)
        if debug_paths or val_count == 0:
            _debug_dataset_path("val", data_path_val, "task_*.pt")
        if debug_paths or train_count == 0:
            _debug_dataset_path("train", data_path_train, "task_*.pt")
        if val_count == 0:
            raise RuntimeError(f"No task_*.pt files found in {data_path_val}")
        if train_count == 0:
            raise RuntimeError(f"No task_*.pt files found in {data_path_train}")
        dset_val = TrajectoryDataset(
            data_path_val,
            task_files=val_files,
            num_samples=val_count,
            num_demos=cfg['num_demos'],
            traj_horizon=cfg['traj_horizon'],
            pred_horizon=cfg['pre_horizon'],
            num_points=num_points,
            rand_g_prob=0.0,
            subsample_live=subsample_live,
            live_spacing_trans=live_spacing_trans,
            live_spacing_rot=live_spacing_rot,
        )
        dset = TrajectoryDataset(
            data_path_train,
            task_files=train_files,
            num_samples=train_count,
            num_demos=cfg['num_demos'],
            traj_horizon=cfg['traj_horizon'],
            pred_horizon=cfg['pre_horizon'],
            num_points=num_points,
            rand_g_prob=cfg['randomize_g_prob'],
            subsample_live=subsample_live,
            live_spacing_trans=live_spacing_trans,
            live_spacing_rot=live_spacing_rot,
        )
        dataloader_val = DataLoader(dset_val, batch_size=1, shuffle=False)
        dataloader = DataLoader(dset, batch_size=cfg['batch_size'], drop_last=True, shuffle=True, **loader_kwargs)
    else:
        val_count = len(glob(os.path.join(data_path_val, 'data_*.pt')))
        train_count = len(glob(os.path.join(data_path_train, 'data_*.pt')))
        if debug_paths or val_count == 0:
            _debug_dataset_path("val", data_path_val, "data_*.pt")
        if debug_paths or train_count == 0:
            _debug_dataset_path("train", data_path_train, "data_*.pt")
        if val_count == 0:
            raise RuntimeError(f"No data_*.pt files found in {data_path_val}")
        if train_count == 0:
            raise RuntimeError(f"No data_*.pt files found in {data_path_train}")
        dset_val = RunningDataset(data_path_val, val_count, rand_g_prob=0)
        dataloader_val = DataLoader(dset_val, batch_size=1, shuffle=False)

        dset = RunningDataset(data_path_train, train_count, rand_g_prob=cfg['randomize_g_prob'])
        dataloader = DataLoader(dset, batch_size=cfg['batch_size'], drop_last=True, shuffle=True, **loader_kwargs)
    ####################################################################################################################
    logger = None
    if record:
        if use_wandb:
            wandb_kwargs = {
                'project': 'Instant Policy',
                'name': f'{run_name}',
                'save_dir': save_dir,
                'log_model': False,
                'resume': args.wandb_resume,
            }
            if args.wandb_id is not None:
                wandb_kwargs['id'] = args.wandb_id
            logger = WandbLogger(**wandb_kwargs)
        # Dump config to save_dir
        pickle.dump(cfg, open(f'{save_dir}/config.pkl', 'wb'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(
        enable_checkpointing=False,  # We save the models manually.
        accelerator=cfg['device'],
        devices=trainer_devices,
        strategy=trainer_strategy,
        max_steps=cfg['num_iters'],
        enable_progress_bar=True,
        precision='16-mixed',
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=None,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=1,
        gradient_clip_algorithm='norm',
        callbacks=[lr_monitor],
    )

    fit_ckpt_path = resume_ckpt_path
    if resume_ckpt_path is not None:
        fit_ckpt_path = _align_checkpoint_state_dict_for_model(resume_ckpt_path, model)

    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
        val_dataloaders=dataloader_val,
        ckpt_path=fit_ckpt_path,
    )

    # Save last:
    if record:
        model.save_model(f'{save_dir}/last.pt')
