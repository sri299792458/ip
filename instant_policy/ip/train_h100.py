import argparse
import os
import pickle
import tempfile
import time
from collections import OrderedDict, deque
from glob import glob

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import DataLoader

from ip.configs.base_config import config as base_config
from ip.models.diffusion import GraphDiffusion
from ip.utils.running_dataset import RunningDataset


REFERENCE_BATCH_SIZE = 16
REFERENCE_NUM_ITERS = 2_550_000
LR = 1e-5
WEIGHT_DECAY = 1e-2
PRECISION = "bf16-mixed"
PREFETCH_FACTOR = 2
SAMPLE_CACHE_SIZE = 0
TRAIN_FILE_REFRESH_EVERY = 128
TRAIN_POLL_INTERVAL_SEC = 2.0
MIN_START_BATCHES = 16
MIN_START_ITEMS_FLOOR = 256
WANDB_RESUME_POLICY = "allow"
THROUGHPUT_LOG_EVERY_N_STEPS = 100


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
        numbered.sort(key=lambda item: item[0])
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
        aligned_sd[tgt_key] = src_sd[src_norm[_norm_orig_mod_key(tgt_key)]]

    ckpt["state_dict"] = aligned_sd

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    fd, aligned_path = tempfile.mkstemp(prefix=".resume_aligned_", suffix=".pt", dir=ckpt_dir)
    os.close(fd)
    torch.save(ckpt, aligned_path)
    print(
        "Adjusted checkpoint state_dict keys for current model format:\n"
        f"  source={ckpt_path}\n"
        f"  aligned={aligned_path}"
    )
    return aligned_path


def _derived_num_iters(batch_size: int) -> int:
    return max(1, REFERENCE_NUM_ITERS * REFERENCE_BATCH_SIZE // int(batch_size))


def _derived_min_train_items(batch_size: int) -> int:
    return max(MIN_START_ITEMS_FLOOR, MIN_START_BATCHES * int(batch_size))


class H100GraphDiffusion(GraphDiffusion):
    # In the streaming setup, Lightning epochs are artificial. Saving last.pt at every
    # epoch end causes too many checkpoint rewrites, so only save periodic step ckpts
    # and the final last.pt on train end.
    def on_train_epoch_end(self, *args, **kwargs):
        return

    def on_train_end(self):
        if self.record:
            self.save_model(f"{self.save_dir}/last.pt", save_compiled=True)


class ThroughputMonitor(Callback):
    def __init__(self, window_size: int = 50, warmup_steps: int = 20):
        self.window = deque(maxlen=int(window_size))
        self.warmup_steps = int(warmup_steps)
        self._last_time = None

    def on_train_start(self, trainer, pl_module):
        self._last_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._last_time is None:
            self._last_time = time.perf_counter()
            return

        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0:
            return

        global_step = int(trainer.global_step)
        batch_size = int(batch.actions.shape[0])
        samples_per_sec = batch_size / dt
        self.window.append(samples_per_sec)

        if global_step < self.warmup_steps:
            return
        if global_step % THROUGHPUT_LOG_EVERY_N_STEPS != 0:
            return

        mean_samples_per_sec = float(np.mean(self.window))
        metrics = {
            "Train_SamplesPerSec": mean_samples_per_sec,
            "Train_StepsPerSec": mean_samples_per_sec / max(batch_size, 1),
        }
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=global_step)
        print(
            f"[THROUGHPUT] step={global_step} "
            f"samples_per_sec={metrics['Train_SamplesPerSec']:.2f} "
            f"steps_per_sec={metrics['Train_StepsPerSec']:.2f}"
        )


class StreamingBufferDataset(IterableDataset):
    def __init__(self, data_dir: str, min_items: int, rand_g_prob: float = 0.0):
        super().__init__()
        self.data_dir = data_dir
        self.min_items = int(max(1, min_items))
        self.rand_g_prob = float(rand_g_prob)
        self.required_attrs = ["actions", "actions_grip"]

    def _list_files(self):
        return sorted(glob(os.path.join(self.data_dir, "data_*.pt")))

    def _validate_data(self, data):
        for attr in self.required_attrs:
            assert hasattr(data, attr)

    def __iter__(self):
        worker = get_worker_info()
        seed = torch.initial_seed() % (2 ** 32)
        if worker is not None:
            seed += worker.id
        rng = np.random.default_rng(seed)
        cache = OrderedDict()
        files = []
        yielded = 0

        while True:
            if yielded % TRAIN_FILE_REFRESH_EVERY == 0 or len(files) < self.min_items:
                files = self._list_files()
            if len(files) < self.min_items:
                time.sleep(TRAIN_POLL_INTERVAL_SEC)
                continue

            path = files[int(rng.integers(len(files)))]
            try:
                if SAMPLE_CACHE_SIZE <= 0:
                    data = torch.load(path)
                else:
                    mtime = os.path.getmtime(path)
                    cached = cache.get(path)
                    if cached is not None and cached[0] == mtime:
                        cache.move_to_end(path)
                        data = cached[1]
                    else:
                        data = torch.load(path)
                        cache[path] = (mtime, data)
                        cache.move_to_end(path)
                        while len(cache) > SAMPLE_CACHE_SIZE:
                            cache.popitem(last=False)

                self._validate_data(data)
                if rng.random() < self.rand_g_prob:
                    data = data.clone()
                    data.current_grip *= -1
                yield data
                yielded += 1
            except Exception:
                files = []
                time.sleep(min(TRAIN_POLL_INTERVAL_SEC, 1.0))


def _build_parser():
    parser = argparse.ArgumentParser(description="Minimal streaming H100 training entrypoint for Instant Policy.")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--scene_encoder_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--num_iters_override", type=int, default=None)
    return parser


def _load_cfg(args, save_dir, resume_ckpt_path):
    if resume_ckpt_path is not None:
        resume_cfg_path = os.path.join(os.path.dirname(resume_ckpt_path), "config.pkl")
        if os.path.isfile(resume_cfg_path):
            cfg = pickle.load(open(resume_cfg_path, "rb"))
        else:
            cfg = dict(base_config)
    else:
        cfg = dict(base_config)

    cfg["scene_encoder_path"] = args.scene_encoder_path
    cfg["compile_models"] = True
    cfg["batch_size"] = int(args.batch_size)
    cfg["num_iters"] = (
        int(args.num_iters_override)
        if args.num_iters_override is not None
        else _derived_num_iters(args.batch_size)
    )
    cfg["save_every"] = int(args.save_every)
    cfg["lr"] = LR
    cfg["weight_decay"] = WEIGHT_DECAY
    cfg["save_dir"] = save_dir
    cfg["record"] = True
    return cfg


def main():
    args = _build_parser().parse_args()

    torch.set_float32_matmul_precision("high")

    save_dir = os.path.join(args.save_root, args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    resume_ckpt_path = _latest_resume_checkpoint(save_dir)
    if resume_ckpt_path is None:
        print(f"[RESUME] no checkpoint found in {save_dir}; starting fresh")

    cfg = _load_cfg(args, save_dir, resume_ckpt_path)
    min_train_items = _derived_min_train_items(args.batch_size)
    model = H100GraphDiffusion(cfg).to(cfg["device"])

    train_count = len(glob(os.path.join(args.train_dir, "data_*.pt")))
    use_val = bool(args.val_dir)
    val_count = 0
    if train_count == 0:
        raise RuntimeError(f"No data_*.pt files found in {args.train_dir}")
    if use_val:
        val_count = len(glob(os.path.join(args.val_dir, "data_*.pt")))
        if val_count == 0:
            raise RuntimeError(f"No data_*.pt files found in {args.val_dir}")

    config_summary = (
        f"[TRAIN_CONFIG] train_items={train_count} "
        f"batch_size={cfg['batch_size']} num_iters={cfg['num_iters']} "
        f"precision={PRECISION} min_train_items={min_train_items}"
    )
    if use_val:
        config_summary += f" val_items={val_count}"
    else:
        config_summary += " validation=disabled"
    print(config_summary)
    print(
        f"[TRAIN_CONFIG] lr={cfg['lr']} weight_decay={cfg['weight_decay']} "
        f"save_every={cfg['save_every']} num_workers={args.num_workers}"
    )

    loader_kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": True,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    dset_train = StreamingBufferDataset(
        args.train_dir,
        min_items=min_train_items,
        rand_g_prob=cfg["randomize_g_prob"],
    )
    dataloader_train = DataLoader(
        dset_train,
        batch_size=cfg["batch_size"],
        drop_last=True,
        **loader_kwargs,
    )

    dataloader_val = None
    if use_val:
        val_loader_kwargs = {
            "num_workers": min(4, int(args.num_workers)),
            "pin_memory": True,
        }
        if val_loader_kwargs["num_workers"] > 0:
            val_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

        dset_val = RunningDataset(args.val_dir, val_count, rand_g_prob=0, sample_cache_size=0)
        dataloader_val = DataLoader(dset_val, batch_size=1, shuffle=False, **val_loader_kwargs)

    wandb_kwargs = {
        "project": "Instant Policy",
        "name": args.run_name,
        "save_dir": save_dir,
        "log_model": False,
        "resume": WANDB_RESUME_POLICY,
    }
    if args.wandb_id is not None:
        wandb_kwargs["id"] = args.wandb_id
    logger = WandbLogger(**wandb_kwargs)

    pickle.dump(cfg, open(os.path.join(save_dir, "config.pkl"), "wb"))

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ThroughputMonitor(),
    ]
    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,
        max_steps=cfg["num_iters"],
        enable_checkpointing=False,
        enable_progress_bar=True,
        benchmark=True,
        precision=PRECISION,
        val_check_interval=min(int(args.save_every), cfg["num_iters"]) if use_val else None,
        limit_val_batches=1.0 if use_val else 0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        logger=logger,
        log_every_n_steps=THROUGHPUT_LOG_EVERY_N_STEPS,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
    )

    fit_ckpt_path = resume_ckpt_path
    if resume_ckpt_path is not None:
        fit_ckpt_path = _align_checkpoint_state_dict_for_model(resume_ckpt_path, model)

    fit_kwargs = {
        "model": model,
        "train_dataloaders": dataloader_train,
        "ckpt_path": fit_ckpt_path,
    }
    if use_val:
        fit_kwargs["val_dataloaders"] = dataloader_val
    trainer.fit(**fit_kwargs)


if __name__ == "__main__":
    main()
