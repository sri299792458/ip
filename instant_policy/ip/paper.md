# Instant Policy Paper Quick Reference

Purpose: keep the key paper constants/claims locally in-repo so implementation decisions do not depend on repeated online lookup.

## Canonical Sources

- Local in-repo corpus:
  - `instant_policy/docs/papers/instant_policy/2411.12633v2.pdf`
  - `instant_policy/docs/papers/instant_policy/2411.12633v2.layout.txt`
  - `instant_policy/docs/papers/instant_policy/2411.12633v2.txt`
  - `instant_policy/docs/papers/instant_policy/instant_policy_bimanual.pdf`
  - `instant_policy/docs/papers/instant_policy/instant_policy_bimanual.layout.txt`
  - `instant_policy/docs/papers/instant_policy/instant_policy_bimanual.txt`
- arXiv abstract page: https://arxiv.org/abs/2411.12633
- arXiv PDF (latest version): https://arxiv.org/pdf/2411.12633
- ar5iv HTML (easy to search): https://ar5iv.org/html/2411.12633

## Constants We Treat As Paper-Canonical

- Context demos per sample: `N = 2`
- Waypoints per context demo: `L = 10`
- Prediction horizon: `T = 8`
- Pseudo waypoint count: `2..6`
- Interpolation spacing target: `1 cm`, `3 deg`
- Bias sampling: `50%` common skills, `50%` random
- Disturbance augmentation probability: `30%`
- Gripper-state augmentation probability: `10%`
- Training schedule: `2.5M` optimization steps + `50K` LR cooldown
- Geometry encoder: pre-trained and frozen during policy training

## Local Code Mapping

- Pseudo-generation constants:
  - `instant_policy/ip/generation/config.py`
- Pseudo-generation implementation + paper alignment notes:
  - `instant_policy/ip/generation/README.md`
- Training defaults:
  - `instant_policy/ip/configs/base_config.py`
  - `instant_policy/ip/models/diffusion.py`
- Geometry encoder load/freeze:
  - `instant_policy/ip/models/model.py`

## Notes

- Ring buffer size is an implementation choice (paper requires continuous replacement, not a fixed numeric size).
- For trajectory storage, `BUFFER_SIZE` means number of `task_*.pt` files (tasks), not timestep samples.
- Paper-reported runtime anchor for single-arm model:
  - `2.5M` optimisation steps + `50K` cooldown in ~`5 days` on one `RTX 3080 Ti`.
  - Pseudo-demonstrations are continuously generated in parallel and old samples are replaced.
