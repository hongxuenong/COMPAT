"""
NFPA — Next-Frame Prediction Attack for AI watermark removal.

Reference implementation: https://github.com/1249748036/NFPA
Paper: "The Future Unmarked: Watermark Removal in AI-Generated Images via
        Next-Frame Prediction" (NeurIPS 2025).

The attack is zero-shot and watermark-agnostic:

    1. DDIM-invert the (watermarked) image into the latent space of a pretrained
       Stable Diffusion model (SD 2.1 base by default).
    2. Treat the inverted latent as the *current frame* and synthesize the
       *next frame* by warping the latent with a motion field (``xyz``) and
       running a short denoising pass through ``MyStableDiffusionPipeline``.
    3. The predicted next frame is a clean, watermark-free reconstruction of the
       scene — the high-frequency / spectral watermark signal does not survive
       the inversion + motion-warp + re-denoise round trip.

This module exposes a uniform removal interface compatible with the project's
``eval.py`` pipeline:

    remove_watermark(image_path, out_dir="recon", ...) -> output_path (str)

plus the lower-level ``nfp_attack(image, ...)`` used by the upstream notebook.

The heavy Stable Diffusion pipeline is loaded lazily on first use and cached, so
importing this module is cheap (e.g. for dry runs with the removal step skipped).
"""

import os

# Match the rest of the project's CUDA env conventions without clobbering values
# the user may already have set.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

# Make the vendored upstream ``utils.py`` (MyStableDiffusionPipeline) importable
# regardless of the current working directory.
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from diffusers import DDIMScheduler, DDIMInverseScheduler

# ── Configuration ─────────────────────────────────────────────────────────────
# Pretrained T2I model used for next-frame prediction. Override with the
# NFPA_MODEL (or NFPA_MODEL_PATH) environment variable to use a local checkpoint.
_DEFAULT_MODEL = os.environ.get(
    "NFPA_MODEL",
    os.environ.get("NFPA_MODEL_PATH", "stabilityai/stable-diffusion-2-1-base"),
)

# SD latent space operates on 512x512 for SD 2.1 base; the upstream attack runs
# inversion and generation at this resolution.
_IMG_SIZE = 512

# Default attack hyper-parameters (upstream defaults from nfp_main.ipynb).
NFP_INFERENCE_STEPS = 10   # DDIM inversion / denoising steps
NFP_XY = 40                # motion-field search range for the latent warp

device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if device == "cuda" else torch.float32

# ── Lazy, cached pipeline ──────────────────────────────────────────────────────
_PIPE = None


def _get_pipe():
    """Load ``MyStableDiffusionPipeline`` once and cache it."""
    global _PIPE
    if _PIPE is None:
        from utils import MyStableDiffusionPipeline

        pipe = MyStableDiffusionPipeline.from_pretrained(
            _DEFAULT_MODEL, torch_dtype=_dtype
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        pipe.safety_checker = None
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.unet.requires_grad_(False)
        _PIPE = pipe
        print(f"[NFPA] Loaded MyStableDiffusionPipeline from {_DEFAULT_MODEL} on {device}")
    return _PIPE


# ── Image preprocessing ────────────────────────────────────────────────────────
def _prep_image(image):
    """
    Normalise an input (path / PIL.Image / tensor) into a (1, 3, 512, 512)
    float tensor in [0, 1].
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        t = TF.to_tensor(image.convert("RGB")).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        t = image
        if t.ndim == 3:
            t = t.unsqueeze(0)
        t = t.float()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if t.shape[-2:] != (_IMG_SIZE, _IMG_SIZE):
        t = F.interpolate(t, size=(_IMG_SIZE, _IMG_SIZE), mode="bilinear",
                          align_corners=False, antialias=True)
    return t.clamp(0, 1)


# ── DDIM inversion ─────────────────────────────────────────────────────────────
def _inversion_latents(image, num_inference_steps=NFP_INFERENCE_STEPS):
    """
    DDIM-invert an image into Stable Diffusion latents (upstream
    ``inversion_latents_fun``).
    """
    pipe = _get_pipe()
    img = _prep_image(image).to(device)
    img = img.to(_dtype) * 2 - 1  # [0,1] -> [-1,1]

    latents = pipe.vae.encode(img)["latent_dist"].mean
    latents = latents * pipe.vae.config.scaling_factor

    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inversion_latents = pipe(
        prompt="",
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        custom_timesteps=None,
        latents=latents,
        output_type="latent",
        width=_IMG_SIZE,
        height=_IMG_SIZE,
    ).images
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return inversion_latents


# ── Core attack ────────────────────────────────────────────────────────────────
def nfp_attack(image, num_inference_steps=NFP_INFERENCE_STEPS, xy=NFP_XY,
               return_tensor=False):
    """
    Run the Next-Frame Prediction attack on a single image.

    Args:
        image: input image as a path, PIL.Image, or (C,H,W)/(1,C,H,W) tensor in [0,1].
        num_inference_steps: DDIM inversion / denoising steps.
        xy: motion-field search range for the latent warp.
        return_tensor: if True return a (3,H,W) tensor, else a PIL.Image.

    Returns:
        The attacked (watermark-removed) image — the predicted next frame.
    """
    pipe = _get_pipe()
    inversed_latents = _inversion_latents(image, num_inference_steps=num_inference_steps)

    warped_latents_timestep = torch.tensor([0], dtype=torch.long, device=pipe.device)
    images = pipe(
        prompt="",
        num_images_per_prompt=2,
        latents=inversed_latents,
        xyz=[xy, xy, 0],
        num_inference_steps=num_inference_steps,
        warped_latents_timestep=warped_latents_timestep,
    ).images

    # images[0] is the reconstruction of the current frame; images[1] is the
    # predicted next frame, which is used as the watermark-removed output.
    attacked_image = images[1]
    if return_tensor:
        return transforms.ToTensor()(attacked_image)
    return attacked_image


# ── eval.py-compatible removal interface ───────────────────────────────────────
def remove_watermark(image_path: str, out_dir: str = "recon",
                     num_inference_steps: int = NFP_INFERENCE_STEPS,
                     xy: int = NFP_XY) -> str:
    """
    Remove a watermark from ``image_path`` via the NFPA next-frame-prediction
    attack and save the reconstruction.

    Drop-in replacement for ``compat_flux.remove_watermark`` used by ``eval.py``.

    Returns:
        Absolute path to the saved reconstructed image.
    """
    attacked = nfp_attack(image_path, num_inference_steps=num_inference_steps, xy=xy)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(image_path))
    attacked.save(out_path)
    print(f"[NFPA] Saved: {out_path}")
    return os.path.abspath(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NFPA watermark-removal attack")
    parser.add_argument("image", help="path to the (watermarked) input image")
    parser.add_argument("--out_dir", default="recon")
    parser.add_argument("--steps", type=int, default=NFP_INFERENCE_STEPS)
    parser.add_argument("--xy", type=int, default=NFP_XY)
    args = parser.parse_args()

    out = remove_watermark(args.image, out_dir=args.out_dir,
                           num_inference_steps=args.steps, xy=args.xy)
    print(out)
