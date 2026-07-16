"""
calib.py — Pixel-wise color calibration between an original image and a
diffusion-generated version of the same scene.

Two methods:
  - reinhard  : per-channel mean+std transfer in LAB space (fast, global)
  - histogram  : per-channel histogram matching (handles non-linear shifts)

Usage:
    python calib.py original.jpg generated.jpg --method reinhard --output out.png
    python calib.py original.jpg generated.jpg --method histogram
"""

import argparse
import os

import numpy as np
from PIL import Image


# ── Core calibration methods ──────────────────────────────────────────────────

def reinhard_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Shift target's per-channel mean and std to match source (Reinhard et al.).
    Operates in LAB space for perceptually uniform color transfer.

    Args:
        source: HxWx3 float32 [0,1], the reference (original).
        target: HxWx3 float32 [0,1], the image to correct (generated).

    Returns:
        Corrected image as HxWx3 float32 [0,1].
    """
    import cv2

    def to_lab(img):
        return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    def from_lab(lab):
        lab_u8 = np.clip(lab, [0, -128, -128], [100, 127, 127])
        lab_u8 = lab_u8.astype(np.float32)
        return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    src_lab = to_lab(source)
    tgt_lab = to_lab(target)

    result = np.empty_like(tgt_lab)
    for c in range(3):
        src_mean, src_std = src_lab[..., c].mean(), src_lab[..., c].std()
        tgt_mean, tgt_std = tgt_lab[..., c].mean(), tgt_lab[..., c].std()
        if tgt_std < 1e-6:
            result[..., c] = src_mean
        else:
            result[..., c] = (tgt_lab[..., c] - tgt_mean) * (src_std / tgt_std) + src_mean

    return np.clip(from_lab(result), 0, 1)


def histogram_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Match target's per-channel histogram to source using rank-order mapping.

    Args:
        source: HxWx3 float32 [0,1], reference (original).
        target: HxWx3 float32 [0,1], image to correct (generated).

    Returns:
        Corrected image as HxWx3 float32 [0,1].
    """
    result = np.empty_like(target)
    for c in range(3):
        src_vals, src_counts = np.unique(source[..., c], return_counts=True)
        tgt_vals, tgt_counts = np.unique(target[..., c], return_counts=True)

        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]
        tgt_cdf = np.cumsum(tgt_counts).astype(np.float64)
        tgt_cdf /= tgt_cdf[-1]

        # For each target pixel value, find the source value with the closest CDF
        interp = np.interp(tgt_cdf, src_cdf, src_vals)
        result[..., c] = interp[np.searchsorted(tgt_vals, target[..., c].ravel())].reshape(target[..., c].shape)

    return np.clip(result, 0, 1)


# ── Public API ────────────────────────────────────────────────────────────────

METHODS = {
    "reinhard":  reinhard_transfer,
    "histogram": histogram_match,
}


def calibrate(original_path: str, generated_path: str,
              output_path: str = None, method: str = "reinhard") -> str:
    """
    Calibrate the color of a diffusion-generated image to match the original.

    Args:
        original_path:  Path to the reference (original) image.
        generated_path: Path to the generated image to correct.
        output_path:    Save path. Defaults to <generated_stem>_calib<ext>.
        method:         'reinhard' or 'histogram'.

    Returns:
        output_path (str)
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {list(METHODS)}")

    if output_path is None:
        stem, ext = os.path.splitext(generated_path)
        output_path = f"{stem}_calib{ext or '.png'}"

    source = np.array(Image.open(original_path).convert("RGB"), dtype=np.float32) / 255.0
    target = np.array(Image.open(generated_path).convert("RGB"), dtype=np.float32) / 255.0

    # Resize source to target dimensions if they differ
    if source.shape[:2] != target.shape[:2]:
        h, w = target.shape[:2]
        source = np.array(
            Image.fromarray((source * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS),
            dtype=np.float32
        ) / 255.0

    corrected = METHODS[method](source, target)
    Image.fromarray((corrected * 255).round().astype(np.uint8)).save(output_path)
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Color-calibrate a generated image to match an original.")
    parser.add_argument("original",  help="Reference (original) image path")
    parser.add_argument("generated", help="Diffusion-generated image path")
    parser.add_argument("--output",  default=None, help="Output path (default: <generated>_calib<ext>)")
    parser.add_argument("--method",  default="reinhard", choices=list(METHODS),
                        help="Calibration method (default: reinhard)")
    args = parser.parse_args()

    out = calibrate(args.original, args.generated, args.output, args.method)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
