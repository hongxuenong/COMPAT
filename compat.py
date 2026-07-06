import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers.utils import load_image
from model import Swin2SR

device = torch.device("cuda")

# ── Super-resolution model (local Swin2SR, no HuggingFace dependency) ────────
# Load pretrained weights from a local .pth, or initialize randomly for training.
#
# Option A — from official .pth (https://github.com/mv-lab/swin2sr#model-zoo):
#   sr_model = Swin2SR.from_pretrained("Swin2SR_ClassicalSR_X4_64.pth").to(device).eval()
#
# Option B — one-time HuggingFace conversion (requires `transformers` installed):
#   sr_model = Swin2SR.from_hf("caidas/swin2SR-classical-sr-x4-64").to(device).eval()
#   torch.save({'params': sr_model.state_dict()}, "swin2sr_x4_local.pth")
#
# Option C — random init (for training from scratch):
#   sr_model = Swin2SR().to(device).eval()

_CKPT = "swin2sr_x4_local.pth"
if os.path.exists(_CKPT):
    sr_model = Swin2SR.from_pretrained(_CKPT).to(device).eval()
    print(f"Loaded local checkpoint: {_CKPT}")
else:
    sr_model = Swin2SR().to(device).eval()
    print("No checkpoint found — running with random weights. "
          "See compat.py header for loading options.")


# ── Main function ─────────────────────────────────────────────────────────────

def remove_watermark(image_path: str, out_dir: str = "recon", lr_size: int = 256):
    """
    Remove watermark using downsample + Swin2SR super-resolution.

    Pipeline
    --------
    1. Rescale to lr_size × lr_size  — destroys high-frequency watermark signal.
    2. Swin2SR (4×)                  — reconstructs at lr_size*4 resolution.
    3. Resize back to original (H, W).

    Args:
        image_path : path to the watermarked input image.
        out_dir    : directory to save the output.
        lr_size    : LR resolution fed to Swin2SR (default 256 → 1024 SR output).
    """
    pil_image = load_image(image_path)
    img = TF.to_tensor(pil_image).unsqueeze(0)   # (1, 3, H, W) in [0, 1]
    _, _, H, W = img.shape

    # ── Step 1: rescale to lr_size × lr_size (destroy watermark) ─────────────
    lr = F.interpolate(img, size=(lr_size, lr_size), mode="bilinear",
                       align_corners=False, antialias=True)
    print(f"LR size : {tuple(lr.shape[-2:])}")

    # ── Step 2: Swin2SR → HR pixels ──────────────────────────────────────────
    with torch.no_grad():
        hr = sr_model(lr.to(device)).squeeze(0).float().cpu()
    print(f"SR size : {tuple(hr.shape[-2:])}")

    # Resize back to original dimensions
    if hr.shape[-2] != H or hr.shape[-1] != W:
        hr = F.interpolate(hr.unsqueeze(0), size=(H, W),
                           mode="bilinear", align_corners=False, antialias=True).squeeze(0)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    out_path = os.path.join(out_dir, image_name)
    TF.to_pil_image(hr).save(out_path)
    print(f"Saved   : {out_path}  ({W}×{H})")
    return out_path


if __name__ == "__main__":
    image_path = "test_images/gemini.png"   # <-- change this
    remove_watermark(image_path)
