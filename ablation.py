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
import gc
import importlib
import itertools
import os
import traceback
import sys

from tqdm import tqdm

os.environ.setdefault("CUDA_DEVICE_ORDER",    "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
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
    # ("scale", "scale",   64),
    # ("scale", "scale",   128),
    ("scale", "scale",   192),
    ("scale", "scale",   256),
    # ("blur",  "sigma",   2.0),
    # ("blur",  "sigma",   5.0),
    ("blur",  "sigma",   8.0),
    # ("noise", "std",     0.05),
    # ("noise", "std",     0.1),
    ("noise", "std",     0.5),
    # ("jpeg",  "quality", 30),
    # ("jpeg",  "quality", 60),
    ("jpeg",  "quality", 90),
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

    csv_exists     = os.path.exists(csv_path)
    sum_exists     = os.path.exists(summary_path)
    csv_file       = open(csv_path,     "a", newline="")
    summary_file   = open(summary_path, "a", newline="")
    writer         = csv.DictWriter(csv_file,     fieldnames=FIELDS)
    summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
    if not csv_exists:
        writer.writeheader()
    if not sum_exists:
        summary_writer.writeheader()

    # Build resume set from existing CSV
    done = set()
    if csv_exists:
        with open(csv_path, newline="") as _f:
            for r in csv.DictReader(_f):
                done.add((r.get("wm_method", ""), r.get("image", ""),
                          r.get("degrade_method", ""), r.get("degrade_value", ""),
                          r.get("use_sam2", ""), r.get("use_lbp", "")))

    all_stats = {}

    combos = list(itertools.product(DEGRADE_CONFIGS, USE_SAM2_OPTIONS, USE_LBP_OPTIONS))
    method_dirs   = [(m, imgs) for m, imgs in _iter_method_dirs(test_folder)]
    images_per_method = {m: (imgs[:n_samples] if n_samples else imgs) for m, imgs in method_dirs}
    total_images  = sum(len(v) for v in images_per_method.values())
    total_units   = len(combos) * total_images

    pbar = tqdm(total=total_units, unit="img", dynamic_ncols=True)
    try:
        for combo_idx, ((deg_method, deg_param, deg_val), use_sam2, use_lbp) in enumerate(combos):
            config_tag = f"{deg_method}={deg_val} sam2={use_sam2} lbp={use_lbp}"
            tqdm.write(f"\n[{combo_idx+1}/{len(combos)}] {config_tag}")

            try:
                model = COMPAT(use_sam2=use_sam2, use_lbp=use_lbp)
            except Exception:
                tqdm.write(f"  Cannot init COMPAT: {traceback.format_exc(limit=1).strip()}")
                pbar.update(total_images)
                continue

            recon_base = os.path.join(out_dir, f"{deg_method}_{deg_val}_sam2{use_sam2}_lbp{use_lbp}")

            for wm_method, sample in images_per_method.items():
                wm = wm_objects.get(wm_method)
                if wm is None:
                    pbar.update(len(sample))
                    continue

                recon_dir = os.path.join(recon_base, wm_method)
                os.makedirs(recon_dir, exist_ok=True)

                stats_key = (deg_method, deg_param, deg_val, use_sam2, use_lbp, wm_method)
                if stats_key not in all_stats:
                    all_stats[stats_key] = _Stats()
                stats = all_stats[stats_key]

                for img_path in sample:
                    filename = img_path.name
                    pbar.set_description(f"{config_tag}/{wm_method}/{filename}")

                    done_key = (wm_method, filename, deg_method, str(deg_val),
                                str(use_sam2), str(use_lbp))
                    if done_key in done:
                        pbar.update(1)
                        continue

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
                        del wm_t, recon_t

                        pbar.set_postfix(det=str(atk_detected), psnr=psnr, ssim=ssim)

                    except Exception:
                        err = traceback.format_exc(limit=3)
                        row["error"] = err.replace("\n", " | ")
                        tqdm.write(f"ERROR [{wm_method}] {filename}: {err.splitlines()[-1]}")

                    row.update({"attacked_detected": atk_detected,
                                "attacked_bit_accuracy": atk_ba,
                                "psnr": psnr, "ssim": ssim, "lpips": lpips, "clip_score": clip})
                    stats.update(atk_detected, atk_ba, psnr, ssim, lpips, clip)
                    writer.writerow(row)
                    csv_file.flush()
                    pbar.update(1)

            del model
            gc.collect()
            torch.cuda.empty_cache()

            combo_summaries = [
                all_stats[(deg_method, deg_param, deg_val, use_sam2, use_lbp, wm_method)].summary(
                    deg_method, deg_param, deg_val, use_sam2, use_lbp, wm_method)
                for wm_method in wm_objects
                if (deg_method, deg_param, deg_val, use_sam2, use_lbp, wm_method) in all_stats
            ]
            summary_writer.writerows(combo_summaries)
            summary_file.flush()

    finally:
        pbar.close()
        csv_file.close()
        summary_file.close()

    print(f"\nPer-image results : {csv_path}")
    print(f"Summary           : {summary_path}")
    return list(all_stats.values())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-folder", default=TEST_FOLDER)
    parser.add_argument("--out-dir",      default="ablation_out")
    parser.add_argument("--n-samples",   type=int, default=None,
                        help="max images to sample per watermark folder (default: all)")
    args = parser.parse_args()
    run_ablation(test_folder=args.test_folder, out_dir=args.out_dir, n_samples=args.n_samples)
