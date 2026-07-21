# Collection of Watermark Removal methods (attacks)

Each attack exposes a uniform interface:

```python
remove_watermark(image_path, out_dir="recon", **attack_kwargs) -> output_path
```

and is registered in `eval.py` under a short name (see `ATTACKS` there).

| Attack name | Location | Method | Key kwargs |
|---|---|---|---|
| `compat` | `compat.py` → `COMPAT` (repo root) | Degrade → feature extraction → **Flux2Klein** reconstruction | `degrade_method`, `scale` |
| `compat_omnigen2` | `compat.py` → `COMPAT_omnigen2` (repo root) | Degrade → feature extraction → **OmniGen2** instruction-guided editing | `degrade_method`, `scale` |
| `nfpa` | `attacks/nfpa/` | **Next-Frame Prediction Attack** — DDIM-invert → motion-warp latent → re-denoise | `num_inference_steps=10`, `xy=40` |

## NFPA
- (NeurIPS 2025) [NFPA](https://github.com/1249748036/NFPA) — see `attacks/nfpa/README_INTEGRATION.md`.

## OmniGen2 (instruction-guided editing)
- [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) — `COMPAT_omnigen2` in `compat.py`
  inherits everything from `COMPAT` except model loading and `reconstruct`, which use the
  `OmniGen2Pipeline`. Requires the `omnigen2` package: clone the repo and `pip install -e .`,
  or set `OMNIGEN2_REPO=/path/to/OmniGen2`. Weights default to the `OmniGen2/OmniGen2`
  HuggingFace repo (override with `OMNIGEN2_MODEL`).

Run any of them through the unified evaluation: `python eval.py --attack <name>` (see the
top-level `README.md`).
