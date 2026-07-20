"""
eval_baselines.py — Evaluation for baseline watermark-removal attacks (da, va, nfpa).

TEST_FOLDER layout:

    test_folder/
        dwt_dct/
            img1.jpg
        vine/
            img1.jpg
        ...

Usage:
    python eval_baselines.py --attack da
    python eval_baselines.py --attack va
    python eval_baselines.py --attack nfpa
"""

import os
os.environ.setdefault("CUDA_DEVICE_ORDER",       "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",    "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import importlib
import sys
import traceback
from pathlib import Path

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT   = os.path.dirname(os.path.abspath(__file__))
_WM_DIR = os.path.join(_ROOT, "watermarks")
_ATK_DIR = os.path.join(_ROOT, "attacks")
for _p in (_WM_DIR, _ATK_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metric import MetricEvaluator
_metrics = MetricEvaluator()

# ── Attack registry ───────────────────────────────────────────────────────────
ATTACKS = {
    "nfpa": ("nfpa",             "remove_watermark"),
    "da":   ("WatermarkAttacker", "remove_watermark_diffusion"),
    "va":   ("WatermarkAttacker", "remove_watermark_vae"),
}


def list_attacks():
    return list(ATTACKS.keys())


def load_remover(attack, skip_removal=False):
    if skip_removal or os.environ.get("SKIP_REMOVAL") == "1":
        def _stub(*_):
            raise NotImplementedError("SKIP_REMOVAL mode — removal disabled")
        return _stub
    mod_name, attr = ATTACKS[attack]
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


# ── Config defaults ───────────────────────────────────────────────────────────
TEST_FOLDER = "test_folder"
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── CSV schema ────────────────────────────────────────────────────────────────
FIELDS = [
    "removal_method",
    "wm_method",
    "image",
    "wm_detected",
    "wm_bit_accuracy",
    "attacked_detected",
    "attacked_bit_accuracy",
    "psnr",
    "ssim",
    "lpips",
    "clip_score",
    "error",
]

SUMMARY_FIELDS = [
    "removal_method",
    "wm_method",
    "total",
    "wm_accuracy",
    "avg_wm_bit_accuracy",
    "attacked_accuracy",
    "avg_attacked_bit_accuracy",
    "avg_psnr",
    "avg_ssim",
    "avg_lpips",
    "avg_clip_score",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _get_detected(result):
    if isinstance(result, dict) and "detected" in result:
        return bool(result["detected"])
    return None


def _load_method(name):
    try:
        return importlib.import_module(name)
    except Exception as e:
        tqdm.write(f"[{name}] Cannot import: {e}")
        return None


# ── Accuracy accumulator ─────────────────────────────────────────────────────

class _MethodStats:
    def __init__(self):
        self.total = self.wm_dec = self.wm_det = 0
        self.atk_dec = self.atk_det = 0
        self.wm_ba_sum = self.atk_ba_sum = 0.0
        self.wm_ba_n   = self.atk_ba_n   = 0
        self.psnr_sum = self.ssim_sum = self.lpips_sum = self.clip_sum = 0.0
        self.metric_n = 0

    def update(self, wm_detected, wm_ba, attacked_detected, atk_ba,
               psnr_val, ssim_val, lpips_val, clip_val):
        self.total += 1
        if wm_detected is not None:
            self.wm_dec += 1
            self.wm_det += int(wm_detected)
        if wm_ba != "":
            self.wm_ba_sum += float(wm_ba)
            self.wm_ba_n   += 1
        if attacked_detected is not None:
            self.atk_dec += 1
            self.atk_det += int(attacked_detected)
        if atk_ba != "":
            self.atk_ba_sum += float(atk_ba)
            self.atk_ba_n   += 1
        if psnr_val != "":
            self.psnr_sum  += float(psnr_val)
            self.ssim_sum  += float(ssim_val)
            self.lpips_sum += float(lpips_val)
            self.metric_n  += 1
        if clip_val != "":
            self.clip_sum += float(clip_val)

    def summary(self, removal_name, wm_method):
        def _avg(s, n): return round(s / n, 4) if n else ""
        return {
            "removal_method":            removal_name,
            "wm_method":                 wm_method,
            "total":                     self.total,
            "wm_accuracy":               _avg(self.wm_det,      self.wm_dec),
            "avg_wm_bit_accuracy":       _avg(self.wm_ba_sum,   self.wm_ba_n),
            "attacked_accuracy":         _avg(self.atk_det,     self.atk_dec),
            "avg_attacked_bit_accuracy": _avg(self.atk_ba_sum,  self.atk_ba_n),
            "avg_psnr":                  _avg(self.psnr_sum,    self.metric_n),
            "avg_ssim":                  _avg(self.ssim_sum,    self.metric_n),
            "avg_lpips":                 _avg(self.lpips_sum,   self.metric_n),
            "avg_clip_score":            _avg(self.clip_sum,    self.metric_n),
        }


# ── Main evaluation entry point ───────────────────────────────────────────────

def run_evaluation(attack="da", test_folder=TEST_FOLDER, out_dir=None,
                   csv_path=None, summary_csv=None, attack_kwargs=None,
                   skip_removal=False):
    attack_kwargs = dict(attack_kwargs or {})

    out_dir     = out_dir     or f"out/eval_{attack}_out"
    csv_path    = csv_path    or f"out/eval_{attack}_out/eval_{attack}_results.csv"
    summary_csv = summary_csv or f"out/eval_{attack}_out/eval_{attack}_summary.csv"

    os.makedirs(out_dir, exist_ok=True)
    remove_wm = load_remover(attack, skip_removal=skip_removal)

    csv_exists = os.path.exists(csv_path)
    done = set()
    if csv_exists:
        with open(csv_path, newline="") as _f:
            for r in csv.DictReader(_f):
                done.add((r.get("wm_method", ""), r.get("image", "")))

    csv_file = open(csv_path, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if not csv_exists:
        writer.writeheader()

    all_stats  = {}
    wm_objects = {}

    method_dirs  = [(m, imgs) for m, imgs in _iter_method_dirs(test_folder)]
    total_images = sum(len(imgs) for _, imgs in method_dirs)

    pbar = tqdm(total=total_images, unit="img", dynamic_ncols=True)
    try:
        for wm_method, images in method_dirs:
            mod = _load_method(wm_method)
            if mod is None:
                pbar.update(len(images))
                continue

            if wm_method not in wm_objects:
                try:
                    pbar.set_description(f"Loading {wm_method}")
                    wm_objects[wm_method] = mod.Watermark()
                except Exception:
                    tqdm.write(f"[{wm_method}] Cannot init Watermark: "
                               f"{traceback.format_exc(limit=1).strip()}")
                    wm_objects[wm_method] = None
            wm = wm_objects[wm_method]
            if wm is None:
                pbar.update(len(images))
                continue

            recon_dir = os.path.join(out_dir, wm_method)
            os.makedirs(recon_dir, exist_ok=True)

            if wm_method not in all_stats:
                all_stats[wm_method] = _MethodStats()
            stats = all_stats[wm_method]

            for img_path in images:
                filename = img_path.name
                pbar.set_description(f"{wm_method}/{filename}")

                if (wm_method, filename) in done:
                    pbar.update(1)
                    continue

                row = {f: "" for f in FIELDS}
                row["removal_method"] = attack
                row["wm_method"]      = wm_method
                row["image"]          = filename
                wm_detected = attacked_detected = None
                wm_ba = atk_ba = ""

                recon_path = os.path.join(recon_dir, filename)

                try:
                    verify_wm   = wm.verify_watermark(str(img_path))
                    wm_detected = _get_detected(verify_wm)
                    wm_ba       = verify_wm.get("bit_accuracy", "") if isinstance(verify_wm, dict) else ""
                    row["wm_detected"]    = wm_detected
                    row["wm_bit_accuracy"] = wm_ba

                    if not os.path.exists(recon_path):
                        recon_path = remove_wm(str(img_path), out_dir=recon_dir,
                                               **attack_kwargs)

                    verify_atk      = wm.verify_watermark(recon_path)
                    attacked_detected = _get_detected(verify_atk)
                    atk_ba          = verify_atk.get("bit_accuracy", "") if isinstance(verify_atk, dict) else ""
                    row["attacked_detected"]      = attacked_detected
                    row["attacked_bit_accuracy"]  = atk_ba

                    wm_t    = _to_tensor(Image.open(img_path))
                    recon_t = _to_tensor(Image.open(recon_path).convert("RGB"))
                    if wm_t.shape[-2:] != recon_t.shape[-2:]:
                        recon_t = F.interpolate(recon_t, size=wm_t.shape[-2:],
                                                mode="bilinear", align_corners=False,
                                                antialias=True)

                    row["psnr"]       = round(_metrics.psnr(wm_t, recon_t),       4)
                    row["ssim"]       = round(_metrics.ssim(wm_t, recon_t),       4)
                    row["lpips"]      = round(_metrics.lpips(wm_t, recon_t),      4)
                    row["clip_score"] = round(_metrics.clip_score(wm_t, recon_t), 4)
                    del wm_t, recon_t

                    pbar.set_postfix(
                        det=str(attacked_detected),
                        psnr=row["psnr"],
                        ssim=row["ssim"],
                    )

                except Exception:
                    err = traceback.format_exc(limit=3)
                    row["error"] = err.replace("\n", " | ")
                    tqdm.write(f"ERROR [{wm_method}] {filename}: "
                               f"{err.splitlines()[-1]}")

                stats.update(wm_detected, wm_ba, attacked_detected, atk_ba,
                             row["psnr"], row["ssim"], row["lpips"], row["clip_score"])
                writer.writerow(row)
                csv_file.flush()
                pbar.update(1)

    finally:
        pbar.close()
        csv_file.close()

    summary_rows = [s.summary(attack, wm) for wm, s in sorted(all_stats.items())]

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n{'removal':<12} {'wm_method':<22} {'total':>6} "
          f"{'wm_acc':>8} {'wm_ba':>8} {'atk_acc':>8} {'atk_ba':>8} "
          f"{'avg_psnr':>10} {'avg_ssim':>9} {'avg_lpips':>10} {'avg_clip':>9}")
    print("-" * 116)
    for r in summary_rows:
        print(f"{r['removal_method']:<12} {r['wm_method']:<22} {r['total']:>6} "
              f"{str(r['wm_accuracy']):>8} {str(r['avg_wm_bit_accuracy']):>8} "
              f"{str(r['attacked_accuracy']):>8} {str(r['avg_attacked_bit_accuracy']):>8} "
              f"{str(r['avg_psnr']):>10} {str(r['avg_ssim']):>9} "
              f"{str(r['avg_lpips']):>10} {str(r['avg_clip_score']):>9}")

    print(f"\nPer-image results : {csv_path}")
    print(f"Summary           : {summary_csv}")
    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline watermark-removal evaluation.")
    parser.add_argument("--attack",      default="da", choices=list_attacks())
    parser.add_argument("--test-folder", default=TEST_FOLDER)
    parser.add_argument("--out-dir",     default=None)
    parser.add_argument("--csv",         default=None)
    parser.add_argument("--summary",     default=None)
    parser.add_argument("--skip-removal", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        attack=args.attack,
        test_folder=args.test_folder,
        out_dir=args.out_dir,
        csv_path=args.csv,
        summary_csv=args.summary,
        skip_removal=args.skip_removal,
    )
