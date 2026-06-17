import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

device = "cuda"
dtype = torch.bfloat16

pipeline = Flux2KleinPipeline.from_pretrained(
    "/data/zilin_wang/alc_tasks/video_gen/FLUX.2-klein-4B",
    torch_dtype=dtype,
).to(device)

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers.utils import load_image



def remove_watermark(image_path):
    pil_image = load_image(image_path)  


    # PIL -> tensor [1, 3, H, W], range [0, 1]
    img = TF.to_tensor(pil_image).unsqueeze(0)
    _, _, H, W = img.shape

    # --------------------------------------------------
    # 1. Downsample (e.g. to half resolution)
    # --------------------------------------------------
    scale = 0.25
    img = F.interpolate(img, scale_factor=scale, mode="bilinear",
                        align_corners=False, antialias=True)

    # --------------------------------------------------
    # 2. Add Gaussian noise
    # --------------------------------------------------
    # noise_std = 0.05                      # in [0,1] image units; tune this
    # img = img + noise_std * torch.randn_like(img)
    # img = img.clamp(0, 1)                 # keep valid pixel range

    # --------------------------------------------------
    # 3. Gaussian smoothing (blur)
    # --------------------------------------------------
    # img = TF.gaussian_blur(img, kernel_size=5, sigma=1.0)

    # (optional) back to original size
    # img = F.interpolate(img, size=(H, W), mode="bilinear",
    #                     align_corners=False, antialias=True)

    # tensor -> PIL if you want to inspect/save
    processed_pil = TF.to_pil_image(img[0].clamp(0, 1))

    image = processed_pil
    # print(image.size)
    
    prompt = "denoise the image and make the image clear."

    image_name = image_path.split("/")[-1]
    image = pipeline(
        prompt=prompt,
        image=[image],
        height=H,
        width=W,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]

    image.save(f'recon/{image_name}')
    print(f"Image save to recon/{image_name}!")
    
if __name__ == "__main__":
    image_path = "test_images/gemini.png"   # <-- change this
    remove_watermark(image_path)