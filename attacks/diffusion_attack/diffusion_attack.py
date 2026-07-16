import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

device = "cuda"
dtype  = torch.bfloat16
_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained(
    _MODEL_ID,
    torch_dtype=dtype,
).to(device)
pipeline.set_progress_bar_config(disable=True)
print(f"Loaded StableDiffusion3Img2ImgPipeline from {_MODEL_ID}")


def remove_watermark(
    image_path: str,
    out_dir: str = "recon",
    scale: float = 0.25,
    strength: float = 0.75,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    prompt: str = "denoise the image and make the image clear. Maintain color of the image unchanged, regenerate content by referencing to the second image.",
    seed: int = 0,
) -> str:
    """
    Remove watermark using Stable Diffusion 3 Medium img2img regeneration.

    Strategy:
      1. Downsample by `scale`  — destroys the high-frequency watermark signal.
      2. Bilinear upsample back to original resolution — blurry reference image.
      3. SD3 img2img at `strength` — re-synthesises detail without the watermark.

    Args:
        image_path: Path to the watermarked image.
        out_dir: Directory to save the output image.
        scale: Downsample factor before regeneration (default 0.25).
        strength: Noise strength for img2img (0–1). Higher = more regeneration,
                  less watermark, but more content drift. (default 0.75)
        num_inference_steps: Denoising steps (default 28).
        guidance_scale: CFG scale; SD3 works well at 4.5–7.0 (default 4.5).
        prompt: Optional text prompt to guide generation.
        seed: RNG seed for reproducibility (default 0).

    Returns:
        Absolute path to the saved output image.
    """
    pil_image = load_image(image_path)
    img = TF.to_tensor(pil_image).unsqueeze(0)   # (1, 3, H, W) in [0, 1]
    _, _, H, W = img.shape

    # Downsample to destroy watermark signal
    lr = F.interpolate(img, scale_factor=scale, mode="bilinear",
                       align_corners=False, antialias=True)

    # Upsample back to original resolution — blurry but watermark-free reference
    ref = F.interpolate(lr, size=(H, W), mode="bilinear",
                        align_corners=False, antialias=True)
    ref_pil = TF.to_pil_image(ref[0].clamp(0, 1))

    # SD3 requires dimensions to be multiples of 16
    H16 = (H // 16) * 16
    W16 = (W // 16) * 16
    if H16 != H or W16 != W:
        ref_pil = ref_pil.resize((W16, H16))
    # extract edge feature of pil_image
    from PIL import ImageFilter
    edge_img = pil_image.filter(ImageFilter.FIND_EDGES)
    result = pipeline(
        prompt=prompt,
        image=[ref_pil, edge_img],
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]
    # result = pipeline(
    #         "A cat holding a sign that says hello world",
    #         negative_prompt="",
    #         num_inference_steps=28,
    #         guidance_scale=7.0,
    #     ).images[0]

    # Restore exact original size if we rounded to multiple of 16
    if H16 != H or W16 != W:
        result = result.resize((W, H), Image.LANCZOS)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(image_path))
    result.save(out_path)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    remove_watermark(
        "/data/xuenong_hong/dataset/aigc/watermark_benchmark/watermarked/watermark_anything/000000000776.jpg",
        out_dir="recon_sd3",
        scale=0.25,
        strength=0.5,
    )
