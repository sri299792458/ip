# Language Modality Transfer

This document explains the current language-modality-transfer implementation in `ip`, how it maps to the Instant Policy paper idea, and how to run it end-to-end.

## Goal

Replace demo-context conditioning with language conditioning while keeping the base action policy mostly frozen.

In practice:
- Base policy (`GraphDiffusion` + `AGI`) is loaded from checkpoint.
- A new `LanguageConditionedEncoder` learns to produce a bottleneck compatible with the teacher model.
- At inference, language bottleneck is injected into the frozen decoder path to generate actions.

## Code Map

- Language training entrypoint: `ip/train_language.py`
- Language evaluation entrypoint: `ip/eval_language.py`
- Language encoder module: `ip/models/language_encoder.py`
- Teacher bottleneck hooks: `ip/models/model.py`
  - `get_demo_bottleneck(...)`
  - `forward_from_bottleneck(...)`
- Language templates + text encoding: `ip/utils/language_utils.py`
- Add language embeddings to dataset: `ip/scripts/build_language_dataset.py`
- Hyperparameters: `ip/configs/language_config.py`

## Data Requirements

Language training uses the step-format dataset (`data_*.pt`) and expects each sample to contain:
- regular policy fields (`pos_demos`, `graps_demos`, `pos_obs`, `actions`, ...)
- `lang_emb` (precomputed text embedding)

`lang_emb` is added with:

```bash
python -m ip.scripts.build_language_dataset \
  --data_dir ./data/train/push_button \
  --task_name push_button \
  --device cuda \
  --add_text
```

Notes:
- This path is designed for `data_*.pt` samples, not `task_*.pt` trajectory files.
- Embeddings are generated with Sentence-BERT (`all-mpnet-base-v2` by default).

## Architecture

## 1) Teacher (Frozen)

`train_language.py` loads `GraphDiffusion` checkpoint and freezes it.

Teacher provides:
- target bottleneck from demos (`get_demo_bottleneck`)
- current observation features (scene/gripper node features after local encoder)

## 2) Language Encoder (Trainable)

`LanguageConditionedEncoder` builds a small heterogeneous graph with node types:
- `scene`
- `gripper`
- `language`

It predicts gripper bottleneck features with shape `[B, Ng, H]` (typically `[B, 6, 1024]`).

## 3) Loss

Training objective in `train_language.py`:
- InfoNCE contrastive loss between predicted bottleneck and teacher bottleneck
- L2 (MSE) bottleneck regression

Combined:
- `loss = contrastive_weight * InfoNCE + l2_weight * MSE`

Defaults from `ip/configs/language_config.py`:
- `contrastive_weight=1.0`
- `l2_weight=0.1`
- `temperature=0.07`

## Training Flow

Per batch:
1. Load batch with `lang_emb`.
2. Teacher computes demo bottleneck target (`no_grad`).
3. Teacher local path computes current scene/gripper features (`no_grad`).
4. Language encoder predicts bottleneck from (current features + `lang_emb`).
5. Compute InfoNCE + L2 losses.
6. Update only language encoder parameters.

Core file: `ip/train_language.py`.

## Inference Flow

`ip/eval_language.py`:
1. Load frozen base policy checkpoint.
2. Load trained language encoder checkpoint.
3. Encode instruction text to `lang_emb` (or load from file).
4. Build current observation graph state.
5. Compute language bottleneck (`compute_language_bottleneck`).
6. Run diffusion denoising using `forward_from_bottleneck(...)`.
7. Execute predicted actions in RLBench rollout loop.

Important implementation detail:
- `forward_from_bottleneck` still runs local + conditional encoder path first, then replaces current gripper bottleneck, then runs action decoder.
- This keeps scene/context feature distribution consistent for the frozen decoder.

Core files:
- `ip/eval_language.py`
- `ip/models/model.py`

## Paper Alignment

This implementation follows the paper’s Appendix-J style idea:
- learn a language-conditioned approximation of the context representation
- preserve the pretrained action policy
- transfer modality at the bottleneck instead of retraining the full policy

Practical difference:
- Current code uses `InfoNCE + L2`, which is a reasonable stabilization choice.

## Quick Commands

## 1) Build language-annotated dataset

```bash
python -m ip.scripts.build_language_dataset \
  --data_dir ./data/train/push_button \
  --task_name push_button \
  --device cuda \
  --add_text
```

## 2) Train language encoder

```bash
python -m ip.train_language \
  --model_path ./checkpoints/ip \
  --data_path_train ./data/train/push_button \
  --data_path_val ./data/val/push_button \
  --batch_size 16 \
  --max_steps 100000 \
  --save_dir ./runs_lang \
  --device cuda
```

## 3) Evaluate with single instruction

```bash
python -m ip.eval_language \
  --task_name push_button \
  --model_path ./checkpoints/ip \
  --lang_encoder_path ./runs_lang/lang_encoder_last.pt \
  --lang_text "Press the button." \
  --num_rollouts 10
```

## 4) Evaluate paraphrase robustness

```bash
python -m ip.eval_language \
  --task_name push_button \
  --model_path ./checkpoints/ip \
  --lang_encoder_path ./runs_lang/lang_encoder_last.pt \
  --paraphrase_file ./paraphrases.txt \
  --num_rollouts 5
```

## Dependencies

- Core training stack: `torch`, `torch-geometric`, `lightning`, `diffusers`
- Text embeddings: `sentence-transformers`
- Language evaluation script currently assumes RLBench environment

## Known Constraints

- `eval_language.py` currently assumes language embedding dim `768` and language encoder layers `4` at load time.
- `train_language.py` expects step-format samples with `lang_emb`; keep dataset directory clean and task-specific.
- RLBench dependency is required for rollout evaluation in `eval_language.py`.

## Summary

The language modality transfer path is implemented as a bottleneck-matching extension, not a full-policy rewrite:
- frozen teacher policy
- trainable language bottleneck encoder
- decoder reuse through bottleneck injection

This is the intended clean design for adding language conditioning to the existing Instant Policy stack.
