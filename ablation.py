"""
ablation.py — Ablation study for COMPAT watermark removal.

Runs every combination of COMPAT settings over the watermarked test images,
recording bit_accuracy, detected, PSNR, SSIM, and LPIPS per image.

Ablation axes:
    degrade_method : scale | blur | noise | jpeg
    degrade_param  : method-specific strength value
    use_flux       : True | False
    use_sam2       : True | False

Usage:
    python ablation.py
    python ablation.py --test-folder /path/to/watermarked --out-dir ablation_out
"""

import csv
import importlib
import itertools
import os
import traceback
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER",    "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image

_ROOT   = os.path.dirname(os.path.abspath(__file__))
_WM_DIR = os.path.join(_ROOT, "watermarks")
for _p in (_WM_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compat import COMPAT
from metric import MetricEvaluator

# ── Test configuration ────────────────────────────────────────────────────────

TEST_FOLDER = "/data/xuenong_hong/dataset/aigc/watermark_benchmark/watermarked"
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── Ablation grid ─────────────────────────────────────────────────────────────
# Each entry: (degrade_method, param_name, param_value)
DEGRADE_CONFIGS = [
    ("scale", "scale",   0.125),
    ("scale", "scale",   0.25),
    ("scale", "scale",   0.5),
    ("blur",  "sigma",   2.0),
    ("blur",  "sigma",   5.0),
    ("noise", "std",     0.05),
    ("noise", "std",     0.1),
    ("jpeg",  "quality", 30),
    ("jpeg",  "quality", 60),
]

USE_SAM2_OPTIONS = [True, False]
USE_LBP_OPTIONS  = [True, False]

# ── CSV schema ────────────────────────────────────────────────────────────────

FIELDS = [
    "wm_method", "image",
    "degrade_method", "degrade_param", "degrade_value",
    "use_sam2", "use_lbp",
    "attacked_detected", "attacked_bit_accuracy",
    "psnr", "ssim", "lpips", "clip_score",
    "error",
]

SUMMARY_FIELDS = [
    "degrade_method", "degrade_param", "degrade_value",
    "use_sam2", "use_lbp",
    "wm_method", "total",
    "attacked_accuracy", "avg_attacked_bit_accuracy",
    "avg_psnr", "avg_ssim", "avg_lpips", "avg_clip_score",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

_metrics = MetricEvaluator()


def _iter_method_dirs(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"TEST_FOLDER not found: {root.resolve()}")
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        images = sorted(p for p in d.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if images:
            yield d.name, images


def _to_tensor(pil_img):
    return TF.to_tensor(pil_img.convert("RGB")).unsqueeze(0)


def _load_wm_modules(test_folder):
    """Import one Watermark instance per method sub-directory."""
    wm_objects = {}
    for method_name, _ in _iter_method_dirs(test_folder):
        try:
            mod = importlib.import_module(method_name)
            wm_objects[method_name] = mod.Watermark()
        except Exception:
            print(f"[{method_name}] Cannot init Watermark — skipped\n"
                  f"{traceback.format_exc(limit=2)}")
            wm_objects[method_name] = None
    return wm_objects


def _get_detected(result):
    if isinstance(result, dict) and "detected" in result:
        return bool(result["detected"])
    return None


def _get_bit_accuracy(result):
    if isinstance(result, dict):
        v = result.get("bit_accuracy", "")
        return "" if v == "" else float(v)
    return ""


class _Stats:
    def __init__(self):
        self.total = self.atk_det = self.atk_dec = 0
        self.atk_ba_sum = 0.0; self.atk_ba_n = 0
        self.psnr = self.ssim = self.lpips = self.clip = 0.0
        self.metric_n = 0

    def update(self, atk_detected, atk_ba, psnr, ssim, lpips, clip):
        self.total += 1
        if atk_detected is not None:
            self.atk_dec += 1; self.atk_det += int(atk_detected)
        if atk_ba != "":
            self.atk_ba_sum += atk_ba; self.atk_ba_n += 1
        if psnr != "":
            self.psnr += float(psnr); self.ssim += float(ssim)
            self.lpips += float(lpips); self.metric_n += 1
        if clip != "":
            self.clip += float(clip)

    def summary(self, deg_method, deg_param, deg_val, use_sam2, use_lbp, wm_method):
        def avg(s, n): return round(s / n, 4) if n else ""
        return {
            "degrade_method":            deg_method,
            "degrade_param":             deg_param,
            "degrade_value":             deg_val,
            "use_sam2":                  use_sam2,
            "use_lbp":                   use_lbp,
            "wm_method":                 wm_method,
            "total":                     self.total,
            "attacked_accuracy":         avg(self.atk_det,    self.atk_dec),
            "avg_attacked_bit_accuracy": avg(self.atk_ba_sum, self.atk_ba_n),
            "avg_psnr":                  avg(self.psnr,       self.metric_n),
            "avg_ssim":                  avg(self.ssim,       self.metric_n),
            "avg_lpips":                 avg(self.lpips,      self.metric_n),
            "avg_clip_score":            avg(self.clip,       self.metric_n),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ablation(test_folder=TEST_FOLDER, out_dir="ablation_out", n_samples=None):
    os.makedirs(out_dir, exist_ok=True)
    csv_path     = os.path.join(out_dir, "ablation_results.csv")
    summary_path = os.path.join(out_dir, "ablation_summary.csv")

    wm_objects = _load_wm_modules(test_folder)

    csv_exists = os.path.exists(csv_path)
    csv_file   = open(csv_path, "a", newline="")
    writer     = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if not csv_exists:
        writer.writeheader()

    all_stats = {}   # (deg_method, deg_param, deg_val, use_sam2, use_lbp, wm_method) -> _Stats
    all_summaries = []

    try:
        combos = list(itertools.product(DEGRADE_CONFIGS, USE_SAM2_OPTIONS, USE_LBP_OPTIONS))
        total_combos = len(combos)

        for combo_idx, ((deg_method, deg_param, deg_val), use_sam2, use_lbp) in enumerate(combos):
            config_tag = f"{deg_method}={deg_val} sam2={use_sam2} lbp={use_lbp}"
            print(f"\n{'='*70}")
            print(f"[{combo_idx+1}/{total_combos}] Config: {config_tag}")
            print(f"{'='*70}")

            try:
                model = COMPAT(use_sam2=use_sam2, use_lbp=use_lbp)
            except Exception:
                print(f"  Cannot init COMPAT:\n{traceback.format_exc(limit=2)}")
                continue

            recon_base = os.path.join(out_dir, f"{deg_method}_{deg_val}_sam2{use_sam2}_lbp{use_lbp}")

            for wm_method, images in _iter_method_dirs(test_folder):
                wm = wm_objects.get(wm_method)
                if wm is None:
                    continue

                recon_dir = os.path.join(recon_base, wm_method)
                os.makedirs(recon_dir, exist_ok=True)

                stats_key = (deg_method, deg_param, deg_val, use_sam2, use_lbp, wm_method)
                if stats_key not in all_stats:
                    all_stats[stats_key] = _Stats()
                stats = all_stats[stats_key]

                sample = images[:n_samples] if n_samples else images
                for img_path in sample:
                    filename = img_path.name
                    row = {f: "" for f in FIELDS}
                    row.update({
                        "wm_method": wm_method, "image": filename,
                        "degrade_method": deg_method, "degrade_param": deg_param,
                        "degrade_value": deg_val, "use_sam2": use_sam2, "use_lbp": use_lbp,
                    })

                    atk_detected = atk_ba = psnr = ssim = lpips = clip = ""
                    out_path = os.path.join(recon_dir, filename)

                    try:
                        if not os.path.exists(out_path):
                            out_path = model.remove_watermark(
                                str(img_path), out_dir=recon_dir,
                                degrade_method=deg_method, **{deg_param: deg_val})

                        verify_atk   = wm.verify_watermark(out_path)
                        atk_detected = _get_detected(verify_atk)
                        atk_ba       = _get_bit_accuracy(verify_atk)

                        wm_t    = _to_tensor(Image.open(img_path))
                        recon_t = _to_tensor(Image.open(out_path))
                        if wm_t.shape[-2:] != recon_t.shape[-2:]:
                            recon_t = F.interpolate(recon_t, size=wm_t.shape[-2:],
                                                    mode="bilinear", align_corners=False,
                                                    antialias=True)
                        psnr  = round(_metrics.psnr(wm_t,  recon_t), 4)
                        ssim  = round(_metrics.ssim(wm_t,  recon_t), 4)
                        lpips = round(_metrics.lpips(wm_t, recon_t), 4)
                        clip  = round(_metrics.clip_score(wm_t, recon_t), 4)
                        ba_str = f"{atk_ba:.3f}" if atk_ba != "" else "n/a"
                        print(f"  [{wm_method}] {filename}: "
                              f"atk_det={atk_detected} ba={ba_str} "
                              f"PSNR={psnr} SSIM={ssim} LPIPS={lpips} CLIP={clip}")

                    except Exception:
                        err = traceback.format_exc(limit=3)
                        row["error"] = err.replace("\n", " | ")
                        print(f"  [{wm_method}] {filename} ERROR: {err}")

                    row.update({"attacked_detected": atk_detected,
                                "attacked_bit_accuracy": atk_ba,
                                "psnr": psnr, "ssim": ssim, "lpips": lpips, "clip_score": clip})
                    stats.update(atk_detected, atk_ba, psnr, ssim, lpips, clip)
                    writer.writerow(row)
                    csv_file.flush()

    finally:
        csv_file.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    for (dm, dp, dv, us, ul, wm), stats in sorted(all_stats.items()):
        all_summaries.append(stats.summary(dm, dp, dv, us, ul, wm))

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(all_summaries)

    print(f"\nPer-image results : {csv_path}")
    print(f"Summary           : {summary_path}")
    return all_summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-folder", default=TEST_FOLDER)
    parser.add_argument("--out-dir",      default="ablation_out")
    parser.add_argument("--n-samples",   type=int, default=None,
                        help="max images to sample per watermark folder (default: all)")
    args = parser.parse_args()
    run_ablation(test_folder=args.test_folder, out_dir=args.out_dir, n_samples=args.n_samples)
