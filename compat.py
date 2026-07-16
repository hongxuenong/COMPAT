"""
compat.py — Watermark removal via degrade → reconstruct.

Degrade methods: scale, blur, noise, jpeg

Usage:
    python compat.py image.jpg --degrade scale --scale 0.25
    python compat.py image.jpg --degrade blur  --sigma 3.0
    python compat.py image.jpg --degrade noise --std 0.05
    python compat.py image.jpg --degrade jpeg  --quality 40
"""

import io
import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16

_FLUX_MODEL_PATH = os.environ.get("COMPAT_FLUX_MODEL",
    "/data/zilin_wang/alc_tasks/video_gen/FLUX.2-klein-4B")

_SAM2_CHECKPOINT = os.environ.get("COMPAT_SAM2_CHECKPOINT",
    "/data/xuenong_hong/models/sam2/sam2.1_hiera_large.pt")
_SAM2_CONFIG = os.environ.get("COMPAT_SAM2_CONFIG",
    "configs/sam2.1/sam2.1_hiera_l.yaml")


class COMPAT:
    """Watermark removal via degrade → feature_extraction → reconstruct."""

    DEGRADE_METHODS = ("scale", "blur", "noise", "jpeg")

    def __init__(self, flux_model_path: str = _FLUX_MODEL_PATH,
                 sam2_checkpoint: str = _SAM2_CHECKPOINT, sam2_config: str = _SAM2_CONFIG,
                 use_sam2: bool = True, use_lbp: bool = True):
        from diffusers import Flux2KleinPipeline
        self._flux = Flux2KleinPipeline.from_pretrained(
            flux_model_path, torch_dtype=dtype
        ).to(device)
        print(f"Loaded Flux2KleinPipeline from {flux_model_path}")

        self._use_lbp = use_lbp
        self._mask_generator = None
        if use_sam2:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
            self._mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
            print(f"Loaded SAM2 from {sam2_checkpoint}")

    def degrade_scale(self, image: Image.Image, scale: float = 0.25) -> Image.Image:
        W, H = image.size
        lw, lh = max(1, int(W * scale)), max(1, int(H * scale))
        return image.resize((lw, lh), Image.BICUBIC).resize((W, H), Image.BICUBIC)

    def degrade_blur(self, image: Image.Image, sigma: float = 3.0) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def degrade_noise(self, image: Image.Image, std: float = 0.05) -> Image.Image:
        arr = TF.to_tensor(image)
        return TF.to_pil_image((arr + torch.randn_like(arr) * std).clamp(0, 1))

    def degrade_jpeg(self, image: Image.Image, quality: int = 40) -> Image.Image:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    def degrade(self, image: Image.Image, method: str = "scale", **kwargs) -> Image.Image:
        fn = {"scale": self.degrade_scale, "blur": self.degrade_blur,
              "noise": self.degrade_noise, "jpeg": self.degrade_jpeg}.get(method)
        if fn is None:
            raise ValueError(f"Unknown degrade method {method!r}. Choose from: {self.DEGRADE_METHODS}")
        return fn(image.convert("RGB"), **kwargs)

    def feature_extraction_lbp(self, image: Image.Image) -> Image.Image:
        """Compute Local Binary Pattern texture map."""
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        padded = np.pad(gray.astype(np.float32), 1, mode="edge")
        lbp = np.zeros((h, w), dtype=np.uint8)
        for bit, (dy, dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
            neighbor = padded[1+dy:h+1+dy, 1+dx:w+1+dx]
            lbp |= (gray >= neighbor).astype(np.uint8) << bit
        return Image.fromarray(cv2.cvtColor(lbp, cv2.COLOR_GRAY2RGB))

    def feature_extraction_segmentation(self, image: Image.Image) -> Image.Image:
        """Run SAM2 automatic mask generation and return a colored segmentation map."""
        import numpy as np

        img_array = np.array(image.convert("RGB"))
        masks = self._mask_generator.generate(img_array)

        seg_map = np.zeros_like(img_array)
        rng = np.random.default_rng(42)
        for mask in sorted(masks, key=lambda m: m["area"]):
            color = rng.integers(50, 230, size=3, dtype=np.uint8)
            seg_map[mask["segmentation"]] = color

        return Image.fromarray(seg_map)

    def feature_extraction(self, image: Image.Image) -> list:
        features = [image.filter(ImageFilter.FIND_EDGES)]
        if self._use_lbp:
            features.append(self.feature_extraction_lbp(image))
        if self._mask_generator is not None:
            features.append(self.feature_extraction_segmentation(image))
        return features

    def reconstruct(self, image_d: Image.Image, target_size: tuple,
                    feature_list: list = []) -> Image.Image:
        W, H = target_size
        resized = image_d.convert("RGB").resize((W, H), Image.LANCZOS)
        img_t = TF.to_tensor(resized).unsqueeze(0)
        H_pad = (16 - H % 16) % 16
        W_pad = (16 - W % 16) % 16
        if H_pad or W_pad:
            img_t = F.pad(img_t, (0, W_pad, 0, H_pad), value=0.0)

        result = self._flux(
            prompt="Denoise the first image and make the image clear. Maintain color of the image unchanged, regenerate content by referencing to the second image till the last image.",
            image=[TF.to_pil_image(img_t[0].clamp(0, 1))] + feature_list,
            height=img_t.shape[-2],
            width=img_t.shape[-1],
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.Generator(device=device).manual_seed(0),
        ).images[0]

        if H_pad or W_pad:
            result = result.crop((0, 0, W, H))
        return result

    def remove_watermark(self, image_path: str, out_dir: str = "recon",
                         degrade_method: str = "scale", **degrade_kwargs) -> str:
        pil = Image.open(image_path).convert("RGB")
        W, H = pil.size

        degraded     = self.degrade(pil, method=degrade_method, **degrade_kwargs)
        feature_list = self.feature_extraction(pil)
        result       = self.reconstruct(degraded, target_size=(W, H), feature_list=feature_list)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(image_path))
        result.save(out_path)
        print(f"Saved: {out_path}")
        return out_path


# Module-level shim so eval.py can call compat.remove_watermark()
def remove_watermark(image_path: str, out_dir: str = "recon",
                     degrade_method: str = "scale", **kwargs) -> str:
    return COMPAT().remove_watermark(image_path, out_dir=out_dir,
                                     degrade_method=degrade_method, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out-dir",  default="recon")
    parser.add_argument("--degrade",  default="scale", choices=COMPAT.DEGRADE_METHODS,
                        dest="degrade_method")
    parser.add_argument("--use-sam2",  action="store_true")
    parser.add_argument("--use-lbp",   action="store_true")
    parser.add_argument("--scale",    type=float, default=0.25)
    parser.add_argument("--sigma",    type=float, default=3.0)
    parser.add_argument("--std",      type=float, default=0.05)
    parser.add_argument("--quality",  type=int,   default=40)
    args = parser.parse_args()

    degrade_kwargs = {"scale": {"scale": args.scale}, "blur": {"sigma": args.sigma},
                      "noise": {"std": args.std},      "jpeg": {"quality": args.quality},
                      }[args.degrade_method]

    COMPAT(use_sam2=args.use_sam2, use_lbp=args.use_lbp).remove_watermark(
        args.image, out_dir=args.out_dir, degrade_method=args.degrade_method, **degrade_kwargs)
