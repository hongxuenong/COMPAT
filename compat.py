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
import sys

import cv2
import numpy as np
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

# ── OmniGen2 (instruction-guided image editing) ──────────────────────────────
# HuggingFace repo id (or local path) for the OmniGen2 weights.
_OMNIGEN2_MODEL = os.environ.get("OMNIGEN2_MODEL", "OmniGen2/OmniGen2")
# Optional path to a local clone of https://github.com/VectorSpaceLab/OmniGen2
# so that `import omnigen2` resolves without pip-installing the package.
_OMNIGEN2_REPO = os.environ.get("OMNIGEN2_REPO")
# Default negative prompt from the upstream inference.py.
_OMNIGEN2_NEG = (
    "(((deformed))), blurry, over saturation, bad anatomy, disfigured, "
    "poorly drawn face, mutation, mutated, (extra_limb), (ugly), "
    "(poorly drawn hands), fused fingers, messy drawing, broken legs censor, "
    "censored, censor_bar"
)


class COMPAT:
    """Watermark removal via degrade → feature_extraction → reconstruct."""

    DEGRADE_METHODS = ("scale", "blur", "noise", "jpeg")

    def __init__(self, flux_model_path: str = _FLUX_MODEL_PATH,
                 sam2_checkpoint: str = _SAM2_CHECKPOINT, sam2_config: str = _SAM2_CONFIG,
                 use_sam2: bool = True, use_lbp: bool = True,
                 use_find_edges: bool = True):
        from diffusers import Flux2KleinPipeline
        self._flux = Flux2KleinPipeline.from_pretrained(
            flux_model_path, torch_dtype=dtype
        ).to(device)
        print(f"Loaded Flux2KleinPipeline from {flux_model_path}")

        self._init_feature_extractors(
            use_sam2=use_sam2, use_lbp=use_lbp, use_find_edges=use_find_edges,
            sam2_checkpoint=sam2_checkpoint, sam2_config=sam2_config,
        )

    def _init_feature_extractors(self, use_sam2: bool = True, use_lbp: bool = True,
                                 use_find_edges: bool = True,
                                 sam2_checkpoint: str = _SAM2_CHECKPOINT,
                                 sam2_config: str = _SAM2_CONFIG):
        """Set up the (optional) LBP / edge / SAM2 feature extractors.

        Shared by COMPAT and its subclasses so the feature-extraction pipeline is
        identical regardless of which reconstruction backbone is used.
        """
        self._use_lbp = use_lbp
        self._use_find_edges = use_find_edges
        self._mask_generator = None
        if use_sam2:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
            self._mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
            print(f"Loaded SAM2 from {sam2_checkpoint}")

    def degrade_scale(self, image: Image.Image, scale: int = 128) -> Image.Image:
        image = TF.to_tensor(image).unsqueeze(0)
        lr = F.interpolate(image, size=(scale, scale), mode="bilinear",
                       align_corners=False, antialias=True)
        lr_pil = TF.to_pil_image(lr[0].clamp(0, 1))
        return lr_pil

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
        if self._use_find_edges:
            features = [image.filter(ImageFilter.FIND_EDGES)]
        else:
            features = []
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

    @staticmethod
    def calibration(source, target):
        from calib import histogram_match
        source = np.array(source, dtype=np.float32) / 255.0
        target = np.array(target, dtype=np.float32) / 255.0
        corrected = histogram_match(source, target)
        return Image.fromarray((corrected * 255).round().astype(np.uint8))
    def remove_watermark(self, image_path: str, out_dir: str = "recon",
                         degrade_method: str = "scale", **degrade_kwargs) -> str:
        pil = Image.open(image_path).convert("RGB")
        W, H = pil.size

        # Pad to the nearest multiple of 16 with black (zero) pixels
        img = TF.to_tensor(pil).unsqueeze(0)   # (1, 3, H, W) in [0, 1]
        _, _, H, W = img.shape
        H_pad = (16 - H % 16) % 16
        W_pad = (16 - W % 16) % 16
        if H_pad or W_pad:
            img = F.pad(img, (0, W_pad, 0, H_pad), value=0.)
        pil = TF.to_pil_image(img[0].clamp(0, 1))

        degraded     = self.degrade(pil, method=degrade_method, **degrade_kwargs)
        feature_list = self.feature_extraction(pil)
        result       = self.reconstruct(degraded, target_size=(W, H), feature_list=feature_list)

        result = self.calibration(result, pil)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(image_path))
        result.save(out_path)
        print(f"Saved: {out_path}")
        return out_path


class COMPAT_omnigen2(COMPAT):
    """COMPAT variant that reconstructs with the OmniGen2 instruction-guided
    image-editing model instead of Flux2Klein.

    Reference: https://github.com/VectorSpaceLab/OmniGen2

    Everything except model loading and ``reconstruct`` is inherited from
    :class:`COMPAT` (degrade → feature_extraction → reconstruct → calibration).
    Reconstruction is framed as an editing instruction: the degraded image is the
    image to edit, any extracted feature maps are provided as additional reference
    images, and the prompt asks the model to restore a clean version.
    """

    #: Editing instruction used to drive the reconstruction.
    DEFAULT_INSTRUCTION = (
        "Remove any watermark, noise, and compression artifacts from the first "
        "image and restore a clean, sharp, high-quality version of it. Keep the "
        "content, composition, and colors unchanged; use the remaining images as "
        "structural references."
    )

    def __init__(self, model_path: str = _OMNIGEN2_MODEL, weight_dtype=dtype,
                 num_inference_steps: int = 50, text_guidance_scale: float = 5.0,
                 image_guidance_scale: float = 2.0, cfg_range: tuple = (0.0, 1.0),
                 max_sequence_length: int = 1024, negative_prompt: str = _OMNIGEN2_NEG,
                 instruction: str = None, seed: int = 0, cpu_offload: bool = False,
                 use_sam2: bool = False, use_lbp: bool = False,
                 use_find_edges: bool = True,
                 sam2_checkpoint: str = _SAM2_CHECKPOINT,
                 sam2_config: str = _SAM2_CONFIG):
        # ── Model loading (OmniGen2 instead of Flux2Klein) ───────────────────
        self._pipe = self._load_omnigen2(model_path, weight_dtype, cpu_offload)

        # Generation hyper-parameters (see upstream inference.py).
        self._instruction          = instruction or self.DEFAULT_INSTRUCTION
        self._num_inference_steps  = num_inference_steps
        self._text_guidance_scale  = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._cfg_range            = tuple(cfg_range)
        self._max_sequence_length  = max_sequence_length
        self._negative_prompt      = negative_prompt
        self._seed                 = seed

        # Reuse COMPAT's feature-extraction setup unchanged.
        self._init_feature_extractors(
            use_sam2=use_sam2, use_lbp=use_lbp, use_find_edges=use_find_edges,
            sam2_checkpoint=sam2_checkpoint, sam2_config=sam2_config,
        )

    @staticmethod
    def _load_omnigen2(model_path: str, weight_dtype, cpu_offload: bool):
        """Load the OmniGen2 pipeline exactly as in the upstream inference.py."""
        if _OMNIGEN2_REPO and _OMNIGEN2_REPO not in sys.path:
            sys.path.insert(0, _OMNIGEN2_REPO)
        try:
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
            from omnigen2.models.transformers.transformer_omnigen2 import (
                OmniGen2Transformer2DModel,
            )
        except ImportError as e:
            raise ImportError(
                "OmniGen2 is not importable. Clone "
                "https://github.com/VectorSpaceLab/OmniGen2 and either "
                "`pip install -e .` it, or set the OMNIGEN2_REPO environment "
                "variable to the local repo path. Original error: " + str(e)
            )

        pipeline = OmniGen2Pipeline.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
        )
        # The transformer lives in the `transformer` subfolder of the checkpoint.
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

        if cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)
        print(f"Loaded OmniGen2Pipeline from {model_path}")
        return pipeline

    @staticmethod
    def _round_to_multiple(x: int, m: int = 16) -> int:
        return max(m, int(round(x / m)) * m)

    def reconstruct(self, image_d: Image.Image, target_size: tuple,
                    feature_list: list = []) -> Image.Image:
        """Reconstruct via OmniGen2 instruction-guided editing.

        The degraded image is the primary image to edit; any extracted feature
        maps are passed as additional reference images.
        """
        W, H = target_size
        resized = image_d.convert("RGB").resize((W, H), Image.LANCZOS)
        input_images = [resized] + list(feature_list)

        # OmniGen2 generates on a latent grid; use dims that are multiples of 16.
        gen_w = self._round_to_multiple(W)
        gen_h = self._round_to_multiple(H)

        generator = torch.Generator(device=device).manual_seed(self._seed)
        results = self._pipe(
            prompt=self._instruction,
            input_images=input_images,
            width=gen_w,
            height=gen_h,
            num_inference_steps=self._num_inference_steps,
            max_sequence_length=self._max_sequence_length,
            text_guidance_scale=self._text_guidance_scale,
            image_guidance_scale=self._image_guidance_scale,
            cfg_range=self._cfg_range,
            negative_prompt=self._negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        result = results.images[0]
        if result.size != (W, H):
            result = result.resize((W, H), Image.LANCZOS)
        return result


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
