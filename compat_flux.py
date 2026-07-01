import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

device = "cuda"
dtype  = torch.bfloat16

_MODEL_PATH = os.environ.get(
    "COMPAT_FLUX_MODEL",
    "/data/zilin_wang/alc_tasks/video_gen/FLUX.2-klein-4B",
)

pipeline = Flux2KleinPipeline.from_pretrained(
    _MODEL_PATH,
    torch_dtype=dtype,
).to(device)
print(f"Loaded Flux2KleinPipeline from {_MODEL_PATH}")


def remove_watermark(image_path: str, out_dir: str = "recon", scale: float = 0.25) -> str:
    """
    Remove watermark using Flux2KleinPipeline.

    1. Downsample by `scale`   — destroys high-frequency watermark signal.
    2. Flux2KleinPipeline      — reconstructs at original resolution.

    Returns the absolute path to the saved output image.
    """
    pil_image = load_image(image_path)
    img = TF.to_tensor(pil_image).unsqueeze(0)   # (1, 3, H, W) in [0, 1]
    _, _, H, W = img.shape

    lr = F.interpolate(img, scale_factor=scale, mode="bilinear",
                       align_corners=False, antialias=True)
    lr_pil = TF.to_pil_image(lr[0].clamp(0, 1))

    result = pipeline(
        prompt="denoise the image and make the image clear.",
        image=[lr_pil],
        height=H,
        width=W,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device=device).manual_seed(0),
    ).images[0]

    os.makedirs(out_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    out_path = os.path.join(out_dir, image_name)
    result.save(out_path)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    remove_watermark("test_images/gemini.png")