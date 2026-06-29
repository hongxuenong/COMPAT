import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

from model import FluxVAE

device = torch.device("cuda")

# ── Super-resolution model (Swin2SR, no diffusion) ───────────────────────────
# caidas/swin2SR-realworld-sr-x4-64: real-world 4× upscaler
SR_MODEL_ID = "caidas/swin2SR-classical-sr-x4-64"
sr_processor = AutoImageProcessor.from_pretrained(SR_MODEL_ID)
sr_model = Swin2SRForImageSuperResolution.from_pretrained(SR_MODEL_ID).to(device).eval()
print(f"Loaded {SR_MODEL_ID}")

# ── FluxVAE (encode HR pixels → latents → decode) ────────────────────────────
vae = FluxVAE().to(device).eval()
ckpt = hf_hub_download("ai-toolkit/flux2_vae", "ae.safetensors")
vae.load_state_dict(load_file(ckpt, device="cpu"))
print("Loaded ai-toolkit/flux2_vae")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sr_to_tensor(sr_output) -> torch.Tensor:
    """Convert Swin2SR reconstruction output → (1, 3, H, W) float32 in [-1, 1]."""
    # reconstruction: (1, 3, H, W), values in [0, 1] as float
    t = sr_output.reconstruction.squeeze(0).clamp(0, 1).float()
    return (t.unsqueeze(0) * 2.0 - 1.0).to(device)


def _to_pil(tensor: torch.Tensor):
    """(1, 3, H, W) in [-1, 1] → PIL."""
    return TF.to_pil_image(((tensor.squeeze(0).cpu() + 1.0) / 2.0).clamp(0, 1))


# ── Main function ─────────────────────────────────────────────────────────────

def remove_watermark(image_path: str, out_dir: str = "recon", scale: float = 0.25):
    """
    Remove watermark without any diffusion model.

    Pipeline
    --------
    1. Downsample the image by `scale`      — destroys high-frequency watermark.
    2. Swin2SR (4× super-resolution)        — generates HR pixels from LR image.
    3. FluxVAE encode  → latent codes       — compresses HR pixels into the
                                              Flux latent space.
    4. FluxVAE decode  → final HR image     — reconstructs through the Flux
                                              image prior, refining texture.

    For scale=0.25 (4× downsample) and Swin2SR 4× upscaler, the SR output
    matches the original resolution before the VAE encode-decode refines it.

    Args:
        image_path : path to the watermarked input image.
        out_dir    : directory to save the output.
        scale      : downsample factor (default 0.25 → 4× SR restores resolution).
    """
    pil_image = load_image(image_path)
    img = TF.to_tensor(pil_image).unsqueeze(0)   # (1, 3, H, W) in [0, 1]
    _, _, H, W = img.shape

    # ── Step 1: downsample (destroy watermark) ────────────────────────────────
    lr = F.interpolate(img, scale_factor=scale, mode="bilinear",
                       align_corners=False, antialias=True)
    lr_pil = TF.to_pil_image(lr[0].clamp(0, 1))
    print(f"LR size : {lr_pil.size}")

    # ── Step 2: Swin2SR → HR pixels ──────────────────────────────────────────
    inputs = sr_processor(lr_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        sr_out = sr_model(**inputs)
    # sr_out.reconstruction: (1, 3, H*4, W*4) in [0, 1]
    sr_tensor = _sr_to_tensor(sr_out)            # → (1, 3, H, W) in [-1, 1]
    print(f"SR size : {tuple(sr_tensor.shape[-2:])}")

    # ── Step 3 & 4: FluxVAE encode → latents → decode → refined HR ───────────
    # H and W must be divisible by 8 for the VAE; pad if the SR output is not.
    _, _, Hsr, Wsr = sr_tensor.shape
    pad_h = (8 - Hsr % 8) % 8
    pad_w = (8 - Wsr % 8) % 8
    if pad_h or pad_w:
        sr_tensor = F.pad(sr_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        posterior = vae._encode(sr_tensor)
        latents   = posterior.mode               # (1, 32, Hsr/8, Wsr/8)
        hr        = vae._decode(latents)         # (1, 3, Hsr, Wsr) in [-1, 1]

    # Crop back to SR size before any padding
    hr = hr[:, :, :Hsr, :Wsr].clamp(-1, 1)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    out_path = os.path.join(out_dir, image_name)
    _to_pil(hr).save(out_path)
    print(f"Saved   : {out_path}  (original {W}×{H} → output {Wsr}×{Hsr})")
    return out_path


if __name__ == "__main__":
    image_path = "test_images/gemini.png"   # <-- change this
    remove_watermark(image_path)
