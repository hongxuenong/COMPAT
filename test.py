"""
test.py — Single-image attack + watermark verification test.

Usage:
    python test.py <image> --attack compat --wm trustmark
    python test.py <image> --attack compat --wm ssl_watermarking --degrade blur --sigma 5
    python test.py <image> --attack compat --wm dwt_dct --out-dir test_out
    python test.py <image> --attack nfpa   --wm vine

Prints wm_detected, bit_accuracy, PSNR, SSIM, LPIPS before and after the attack.
"""

import argparse
import importlib
import os
import sys
import traceback

os.environ.setdefault("CUDA_DEVICE_ORDER",       "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",    "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

_ROOT   = os.path.dirname(os.path.abspath(__file__))
_WM_DIR = os.path.join(_ROOT, "watermarks")
for _p in (_WM_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WM_METHODS = [
    "dwt_dct", "ssl_watermarking", "trustmark",
    "watermark_anything", "vine", "editguard", "omniguard",
]

ATTACKS = {
    "compat":         "compat",
    "compat_vllm":    "compat_vllm",
    "nfpa":           "nfpa",
}


def _to_tensor(pil_img):
    return TF.to_tensor(pil_img.convert("RGB")).unsqueeze(0)


def _load_recon_tensor(orig_t, recon_path):
    t = _to_tensor(Image.open(recon_path))
    if orig_t.shape[-2:] != t.shape[-2:]:
        t = F.interpolate(t, size=orig_t.shape[-2:],
                          mode="bilinear", align_corners=False, antialias=True)
    return t


def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)


def main():
    parser = argparse.ArgumentParser(description="Single-image attack + verify test.")
    parser.add_argument("image",          help="Path to a watermarked image")
    parser.add_argument("--attack", "-a", required=True,
                        choices=list(ATTACKS), help="Attack module to use")
    parser.add_argument("--wm",    "-w",  required=True,
                        choices=WM_METHODS, help="Watermark method to verify with")
    parser.add_argument("--out-dir",      default="test_out",
                        help="Directory for the reconstructed image (default: test_out)")
    # compat-specific
    parser.add_argument("--degrade",      default="scale",
                        choices=["scale", "blur", "noise", "jpeg"],
                        help="Degrade method (compat only)")
    parser.add_argument("--scale",        type=float, default=0.25)
    parser.add_argument("--sigma",        type=float, default=3.0)
    parser.add_argument("--std",          type=float, default=0.05)
    parser.add_argument("--quality",      type=int,   default=40)
    parser.add_argument("--use-sam2",     action="store_true")
    parser.add_argument("--use-lbp",      action="store_true")
    # nfpa-specific
    parser.add_argument("--steps",        type=int,   default=None,
                        help="num_inference_steps (nfpa)")
    parser.add_argument("--xy",           type=int,   default=None,
                        help="xy crop size (nfpa)")
    # metrics
    parser.add_argument("--no-metrics",   action="store_true",
                        help="Skip PSNR/SSIM/LPIPS computation")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load watermark verifier ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Watermark method : {args.wm}")
    print(f"Attack           : {args.attack}")
    print(f"Image            : {image_path}")
    print(f"{'─'*60}")

    try:
        wm_mod = importlib.import_module(args.wm)
        wm = wm_mod.Watermark()
        print(f"[wm] Loaded {args.wm}")
    except Exception:
        print(f"[wm] Failed to load {args.wm}:\n{traceback.format_exc(limit=3)}")
        sys.exit(1)

    # ── Step 1: verify original ───────────────────────────────────────────────
    print(f"\n[1] Verifying watermark on original image ...")
    try:
        result_orig = wm.verify_watermark(image_path)
        print(f"    {result_orig}")
    except Exception:
        print(f"    ERROR: {traceback.format_exc(limit=2)}")
        result_orig = {}

    # ── Load attack module ────────────────────────────────────────────────────
    print(f"\n[2] Loading attack: {args.attack} ...")
    try:
        atk_mod = importlib.import_module(ATTACKS[args.attack])

        # COMPAT-style class — instantiate once
        import inspect
        remover = None
        for attr in dir(atk_mod):
            cls = getattr(atk_mod, attr)
            if isinstance(cls, type) and hasattr(cls, "remove_watermark"):
                sig = inspect.signature(cls.__init__)
                init_kw = {}
                if "use_sam2" in sig.parameters:
                    init_kw["use_sam2"] = args.use_sam2
                if "use_lbp" in sig.parameters:
                    init_kw["use_lbp"] = args.use_lbp
                remover = cls(**init_kw).remove_watermark
                print(f"    Instantiated {cls.__name__}")
                break
        if remover is None:
            remover = atk_mod.remove_watermark
            print(f"    Using module-level remove_watermark")
    except Exception:
        print(f"    ERROR loading attack:\n{traceback.format_exc(limit=3)}")
        sys.exit(1)

    # ── Build attack kwargs ───────────────────────────────────────────────────
    attack_kwargs = {}
    if args.attack in ("compat", "compat_vllm"):
        attack_kwargs["degrade_method"] = args.degrade
        attack_kwargs.update({
            "scale":   {"scale": args.scale},
            "blur":    {"sigma": args.sigma},
            "noise":   {"std":   args.std},
            "jpeg":    {"quality": args.quality},
        }[args.degrade])
    if args.steps   is not None: attack_kwargs["num_inference_steps"] = args.steps
    if args.xy      is not None: attack_kwargs["xy"] = args.xy

    # ── Step 2: run attack ────────────────────────────────────────────────────
    print(f"\n[3] Running attack ...")
    try:
        recon_path = remover(image_path, out_dir=args.out_dir, **attack_kwargs)
        print(f"    Saved: {recon_path}")
    except Exception:
        print(f"    ERROR during attack:\n{traceback.format_exc(limit=3)}")
        sys.exit(1)

    # ── Step 3: verify reconstruction ────────────────────────────────────────
    print(f"\n[4] Verifying watermark on reconstruction ...")
    try:
        result_recon = wm.verify_watermark(recon_path)
        print(f"    {result_recon}")
    except Exception:
        print(f"    ERROR: {traceback.format_exc(limit=2)}")
        result_recon = {}

    # ── Step 4: image quality metrics ────────────────────────────────────────
    psnr = ssim = lpips = None
    if not args.no_metrics:
        print(f"\n[5] Computing image quality metrics ...")
        try:
            from metric import MetricEvaluator
            M = MetricEvaluator()
            orig_t  = _to_tensor(Image.open(image_path))
            recon_t = _load_recon_tensor(orig_t, recon_path)
            psnr  = round(M.psnr(orig_t,  recon_t), 4)
            ssim  = round(M.ssim(orig_t,  recon_t), 4)
            lpips = round(M.lpips(orig_t, recon_t), 4)
            print(f"    PSNR={psnr} dB  SSIM={ssim}  LPIPS={lpips}")
        except Exception:
            print(f"    ERROR computing metrics:\n{traceback.format_exc(limit=2)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'':30s}  {'before':>10}  {'after':>10}")
    print(f"{'─'*60}")

    def _get(d, k): return d.get(k, "n/a") if isinstance(d, dict) else "n/a"

    print(f"{'detected':<30}  {str(_get(result_orig,  'detected')):>10}  {str(_get(result_recon, 'detected')):>10}")
    print(f"{'bit_accuracy':<30}  {_fmt(_get(result_orig,  'bit_accuracy')):>10}  {_fmt(_get(result_recon, 'bit_accuracy')):>10}")
    if psnr is not None:
        print(f"{'─'*60}")
        print(f"{'PSNR (dB)':<30}  {'':>10}  {_fmt(psnr):>10}")
        print(f"{'SSIM':<30}  {'':>10}  {_fmt(ssim):>10}")
        print(f"{'LPIPS':<30}  {'':>10}  {_fmt(lpips):>10}")
    print(f"{'─'*60}")
    print(f"Reconstruction : {recon_path}")


if __name__ == "__main__":
    main()
