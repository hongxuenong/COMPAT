"""
compat_vllm.py — Watermark removal via degrade → reconstruct (vLLM backend).

Identical pipeline to compat.py but Flux inference runs through vLLM instead
of diffusers. Two modes:

  In-process (default):
      The vLLM LLM engine is loaded inside COMPAT_VLLM.__init__.
      No server required; GPU memory is used in this process.

  Remote server:
      Set COMPAT_VLLM_URL (or --vllm-url) to a running vLLM server that was
      started with:
          vllm serve <model_path> --task generate --port 8000
      Inference goes over HTTP; no GPU memory used in this process.

Environment variables:
    COMPAT_FLUX_MODEL      path or HF repo for the Flux model
    COMPAT_VLLM_URL        URL of a running vLLM server (remote mode)
    COMPAT_SAM2_CHECKPOINT path to SAM2 .pt checkpoint
    COMPAT_SAM2_CONFIG     SAM2 yaml config path

Usage:
    python compat_vllm.py image.jpg --degrade scale --scale 0.25
    python compat_vllm.py image.jpg --degrade blur --sigma 3.0 --vllm-url http://localhost:8000
"""

import base64
import io
import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_FLUX_MODEL_PATH = os.environ.get("COMPAT_FLUX_MODEL",
    "/data/zilin_wang/alc_tasks/video_gen/FLUX.2-klein-4B")
_VLLM_URL = os.environ.get("COMPAT_VLLM_URL", "")

_SAM2_CHECKPOINT = os.environ.get("COMPAT_SAM2_CHECKPOINT",
    "/data/xuenong_hong/models/sam2/sam2.1_hiera_large.pt")
_SAM2_CONFIG = os.environ.get("COMPAT_SAM2_CONFIG",
    "configs/sam2.1/sam2.1_hiera_l.yaml")

_FLUX_PROMPT = (
    "Denoise the first image and make the image clear. "
    "Maintain color of the image unchanged, regenerate content "
    "by referencing to the second image till the last image."
)
_NUM_INFERENCE_STEPS = 4
_GUIDANCE_SCALE = 1.0


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data-URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    raw = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{raw}"


class COMPAT_VLLM:
    """Watermark removal via degrade → feature_extraction → reconstruct (vLLM)."""

    DEGRADE_METHODS = ("scale", "blur", "noise", "jpeg")

    def __init__(self, flux_model_path: str = _FLUX_MODEL_PATH,
                 vllm_url: str = _VLLM_URL,
                 sam2_checkpoint: str = _SAM2_CHECKPOINT,
                 sam2_config: str = _SAM2_CONFIG,
                 use_sam2: bool = False, use_lbp: bool = False):

        self._vllm_url = vllm_url.rstrip("/") if vllm_url else ""
        self._flux_model_name = os.path.basename(flux_model_path.rstrip("/"))

        if self._vllm_url:
            # Remote mode — communicate with a running vLLM server over HTTP
            from openai import OpenAI
            self._client = OpenAI(
                base_url=f"{self._vllm_url}/v1",
                api_key="EMPTY",
            )
            self._engine = None
            print(f"Using vLLM server at {self._vllm_url}  model={self._flux_model_name}")
        else:
            # In-process mode — load vLLM engine directly
            from vllm import LLM
            self._engine = LLM(
                model=flux_model_path,
                task="generate",
                dtype="bfloat16",
                max_num_seqs=1,
            )
            self._client = None
            print(f"Loaded Flux model via vLLM (in-process) from {flux_model_path}")

        self._use_lbp = use_lbp
        self._mask_generator = None
        if use_sam2:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
            self._mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
            print(f"Loaded SAM2 from {sam2_checkpoint}")

    # ── Degrade methods ───────────────────────────────────────────────────────

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
            raise ValueError(
                f"Unknown degrade method {method!r}. Choose from: {self.DEGRADE_METHODS}")
        return fn(image.convert("RGB"), **kwargs)

    # ── Feature extraction ────────────────────────────────────────────────────

    def feature_extraction_lbp(self, image: Image.Image) -> Image.Image:
        """Compute Local Binary Pattern texture map (8-neighbor, no skimage)."""
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        padded = np.pad(gray.astype(np.float32), 1, mode="edge")
        lbp = np.zeros((h, w), dtype=np.uint8)
        for bit, (dy, dx) in enumerate([
                (-1, -1), (-1, 0), (-1, 1), (0, 1),
                (1, 1),  (1, 0),  (1, -1), (0, -1)]):
            neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            lbp |= (gray >= neighbor).astype(np.uint8) << bit
        return Image.fromarray(cv2.cvtColor(lbp, cv2.COLOR_GRAY2RGB))

    def feature_extraction_segmentation(self, image: Image.Image) -> Image.Image:
        """SAM2 automatic masks → colored RGB segmentation map."""
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

    # ── Flux inference (vLLM) ─────────────────────────────────────────────────

    def _run_flux_inprocess(self, images: list,
                            height: int, width: int) -> Image.Image:
        """In-process vLLM generate call."""
        from vllm import SamplingParams

        outputs = self._engine.generate(
            {
                "prompt": _FLUX_PROMPT,
                "multi_modal_data": {"image": images},
            },
            SamplingParams(
                max_tokens=1,
                guidance_scale=_GUIDANCE_SCALE,
                num_inference_steps=_NUM_INFERENCE_STEPS,
            ),
        )
        return outputs[0].outputs[0].image

    def _run_flux_remote(self, images: list,
                         height: int, width: int) -> Image.Image:
        """
        HTTP call to a vLLM OpenAI-compatible server.

        Images are sent as base64 image_url content blocks (multimodal chat format).
        Diffusion parameters are forwarded via extra_body.
        The response is expected to carry the result image as a base64 data-URI.
        """
        content = [
            {"type": "image_url", "image_url": {"url": _pil_to_b64(img)}}
            for img in images
        ]
        content.append({"type": "text", "text": _FLUX_PROMPT})

        response = self._client.chat.completions.create(
            model=self._flux_model_name,
            messages=[{"role": "user", "content": content}],
            extra_body={
                "guidance_scale":      _GUIDANCE_SCALE,
                "num_inference_steps": _NUM_INFERENCE_STEPS,
                "height":              height,
                "width":               width,
            },
        )

        b64_data = response.choices[0].message.content
        if b64_data.startswith("data:"):
            b64_data = b64_data.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")

    # ── Reconstruct ───────────────────────────────────────────────────────────

    def reconstruct(self, image_d: Image.Image, target_size: tuple,
                    feature_list: list = []) -> Image.Image:
        W, H = target_size
        resized = image_d.convert("RGB").resize((W, H), Image.LANCZOS)
        img_t = TF.to_tensor(resized).unsqueeze(0)

        # Pad to multiples of 16 (required by Flux attention)
        H_pad = (16 - H % 16) % 16
        W_pad = (16 - W % 16) % 16
        if H_pad or W_pad:
            img_t = F.pad(img_t, (0, W_pad, 0, H_pad), value=0.0)

        padded_img = TF.to_pil_image(img_t[0].clamp(0, 1))
        ph, pw = img_t.shape[-2], img_t.shape[-1]
        images = [padded_img] + feature_list

        if self._vllm_url:
            result = self._run_flux_remote(images, height=ph, width=pw)
        else:
            result = self._run_flux_inprocess(images, height=ph, width=pw)

        if H_pad or W_pad:
            result = result.crop((0, 0, W, H))
        return result

    # ── Public entry point ────────────────────────────────────────────────────

    def remove_watermark(self, image_path: str, out_dir: str = "recon",
                         degrade_method: str = "scale", **degrade_kwargs) -> str:
        pil = Image.open(image_path).convert("RGB")
        W, H = pil.size

        degraded     = self.degrade(pil, method=degrade_method, **degrade_kwargs)
        feature_list = self.feature_extraction(pil)
        result       = self.reconstruct(degraded, target_size=(W, H),
                                        feature_list=feature_list)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(image_path))
        result.save(out_path)
        print(f"Saved: {out_path}")
        return out_path


# Module-level shim for eval.py compatibility
def remove_watermark(image_path: str, out_dir: str = "recon",
                     degrade_method: str = "scale", **kwargs) -> str:
    return COMPAT_VLLM().remove_watermark(
        image_path, out_dir=out_dir, degrade_method=degrade_method, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out-dir",   default="recon")
    parser.add_argument("--degrade",   default="scale",
                        choices=COMPAT_VLLM.DEGRADE_METHODS, dest="degrade_method")
    parser.add_argument("--vllm-url",  default=_VLLM_URL,
                        help="URL of a running vLLM server (omit for in-process)")
    parser.add_argument("--use-sam2",  action="store_true")
    parser.add_argument("--use-lbp",   action="store_true")
    parser.add_argument("--scale",     type=float, default=0.25)
    parser.add_argument("--sigma",     type=float, default=3.0)
    parser.add_argument("--std",       type=float, default=0.05)
    parser.add_argument("--quality",   type=int,   default=40)
    args = parser.parse_args()

    degrade_kwargs = {
        "scale": {"scale": args.scale},
        "blur":  {"sigma": args.sigma},
        "noise": {"std":   args.std},
        "jpeg":  {"quality": args.quality},
    }[args.degrade_method]

    COMPAT_VLLM(
        use_sam2=args.use_sam2,
        use_lbp=args.use_lbp,
        vllm_url=args.vllm_url,
    ).remove_watermark(
        args.image, out_dir=args.out_dir,
        degrade_method=args.degrade_method, **degrade_kwargs)
