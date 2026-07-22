"""
compat_omnigen2.py — OmniGen2 instruction-guided reconstruction attack.

Defines ``COMPAT_omnigen2``, a COMPAT variant that reconstructs with the
OmniGen2 instruction-guided image-editing model instead of Flux2Klein.

Reference: https://github.com/VectorSpaceLab/OmniGen2

Everything except model loading and ``reconstruct`` is inherited from
:class:`compat.COMPAT` (degrade → feature_extraction → reconstruct → calibration).

Select it from the evaluation harness, e.g.:

    python eval.py --attack compat_omnigen2
"""

import os
import sys

import torch
from PIL import Image

from compat import COMPAT, device, dtype, _SAM2_CHECKPOINT, _SAM2_CONFIG

# ── OmniGen2 (instruction-guided image editing) ──────────────────────────────
# HuggingFace repo id (or local path) for the OmniGen2 weights.
_OMNIGEN2_MODEL = os.environ.get("OMNIGEN2_MODEL", "OmniGen2/OmniGen2")
# Optional path to a local clone of https://github.com/VectorSpaceLab/OmniGen2
# so that `import omnigen2` resolves without pip-installing the package.
_OMNIGEN2_REPO = os.environ.get("OMNIGEN2_REPO", "/data/zilin_wang/watermark_project/OmniGen2")
# Default negative prompt from the upstream inference.py.
_OMNIGEN2_NEG = (
    "(((deformed))), blurry, over saturation, bad anatomy, disfigured, "
    "poorly drawn face, mutation, mutated, (extra_limb), (ugly), "
    "(poorly drawn hands), fused fingers, messy drawing, broken legs censor, "
    "censored, censor_bar"
)


class COMPAT_omnigen2(COMPAT):
    """COMPAT variant that reconstructs with the OmniGen2 instruction-guided
    image-editing model instead of Flux2Klein.

    Reference: https://github.com/VectorSpaceLab/OmniGen2

    Everything except model loading and ``reconstruct`` is inherited from
    :class:`compat.COMPAT` (degrade → feature_extraction → reconstruct → calibration).
    Reconstruction is framed as an editing instruction: the degraded image is the
    image to edit, any extracted feature maps are provided as additional reference
    images, and the prompt asks the model to restore a clean version.
    """

    #: Editing instruction used to drive the reconstruction.
    DEFAULT_INSTRUCTION = (
        "denoise the image and make the image clear. Keep the content and color of the image unchanged"
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


# Module-level shim so eval.py can call compat_omnigen2.remove_watermark()
def remove_watermark(image_path: str, out_dir: str = "recon",
                     degrade_method: str = "scale", **kwargs) -> str:
    return COMPAT_omnigen2().remove_watermark(
        image_path, out_dir=out_dir, degrade_method=degrade_method, **kwargs)
