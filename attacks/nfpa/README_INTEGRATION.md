# NFPA — Next-Frame Prediction Attack (watermark removal)

Source: https://github.com/1249748036/NFPA
Paper: "The Future Unmarked: Watermark Removal in AI-Generated Images via Next-Frame Prediction" (NeurIPS 2025).

## Files
- `utils.py` — vendored verbatim from upstream (byte-identical, git blob `b2e7a7f0…`).
  Provides `MyStableDiffusionPipeline`, the modified Stable Diffusion pipeline with
  latent motion-warping / next-frame prediction. (The upstream `TreeRingWatermark`
  class also lives here but is not used by this integration.)
- `nfpa_attack.py` — wrapper exposing the project's removal interface:
  - `remove_watermark(image_path, out_dir="recon", num_inference_steps=10, xy=40) -> out_path`
    (drop-in replacement for `compat_flux.remove_watermark`)
  - `nfp_attack(image, num_inference_steps=10, xy=40, return_tensor=False)` — lower-level attack.
- `__init__.py` — exports `remove_watermark`, `nfp_attack`.

## How it works
1. DDIM-invert the watermarked image into Stable Diffusion latents.
2. Warp the latent with a motion field (`xyz=[xy, xy, 0]`) and run a short denoise pass,
   predicting the "next frame" — a clean, watermark-free reconstruction.

The attack is zero-shot and watermark-agnostic (no knowledge of the watermark scheme).

## Model / requirements
- Uses `stabilityai/stable-diffusion-2-1-base` by default (downloaded from HuggingFace on
  first use). Override with `NFPA_MODEL` / `NFPA_MODEL_PATH` (e.g. a local checkpoint path).
- A CUDA GPU is recommended (>= ~24 GB VRAM per the upstream README). Runs at 512×512.
- Deps: `torch`, `torchvision`, `diffusers>=0.31`, `transformers`, `peft`, `Pillow` (see upstream `requirements.txt`).

## Evaluation
`eval_nfpa.py` (project root) runs the standard removal evaluation using this attack:
verify watermark → NFPA removal → re-verify on reconstruction → PSNR/SSIM, with per-image
and per-method CSV summaries. See its module docstring for the `test_folder/<method>/<images>`
layout and env options.
