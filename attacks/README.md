# Collection of Watermark Removal methods (attacks)

Each attack exposes a uniform interface:

```python
remove_watermark(image_path, out_dir="recon", **attack_kwargs) -> output_path
```

and is registered in `eval.py` under a short name (see `ATTACKS` there).

| Attack name | Location | Method | Key kwargs |
|---|---|---|---|
| `compat` | `compat_flux.py` (repo root) | Downsample → **Flux2Klein** diffusion reconstruction | `scale=0.25` |
| `compat_sr` | `compat.py` (repo root) | Downsample → **Swin2SR** 4× → **FluxVAE** encode/decode (no diffusion) | `scale=0.25` |
| `nfpa` | `attacks/nfpa/` | **Next-Frame Prediction Attack** — DDIM-invert → motion-warp latent → re-denoise | `num_inference_steps=10`, `xy=40` |

## NFPA
- (NeurIPS 2025) [NFPA](https://github.com/1249748036/NFPA) — see `attacks/nfpa/README_INTEGRATION.md`.

Run any of them through the unified evaluation: `python eval.py --attack <name>` (see the
top-level `README.md`).
