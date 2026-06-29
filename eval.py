"""
eval.py — Watermark removal evaluation.

TEST_FOLDER layout (images are already watermarked):

    test_folder/
        dwt_dct/
            img1.jpg
            img2.png
        vine/
            img1.jpg
        ...

For every method sub-directory and every image inside it:
  1. Verify watermark              → wm_detected
  2. Attack: remove watermark      → <OUT_DIR>/<method>/<filename>
  3. Verify on reconstruction      → attacked_detected
  4. Compute PSNR / SSIM          (watermarked vs reconstruction)

Results are written row-by-row to CSV_PATH so partial runs are preserved.

Usage:
    python eval.py
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import importlib
import sys
import traceback
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

# Make watermark sub-packages importable as top-level names (dwt_dct, vine, ...)
_WM_DIR = os.path.join(_ROOT, "watermarks")
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
TEST_FOLDER = "test_folder"      # root containing <method>/<images> sub-dirs
OUT_DIR     = "eval_out"         # root output directory
CSV_PATH    = "eval_results.csv" # results table

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── CSV schema ────────────────────────────────────────────────────────────────
FIELDS = [
    "image",            # filename (not full path)
    "method",           # watermark method name
    "wm_detected",      # watermark present in watermarked image (sanity check)
    "attacked_detected",# watermark present after removal attack
    "psnr",             # dB, watermarked vs reconstruction
    "ssim",             # [−1,1], watermarked vs reconstruction
    "error",            # non-empty if any step raised an exception
]

SUMMARY_CSV   = "eval_summary.csv"
SUMMARY_FIELDS = [
    "method",
    "total",             # images processed without fatal error
    "wm_accuracy",       # fraction detected before attack  (higher = embedding works)
    "attacked_accuracy", # fraction detected after attack   (lower  = attack works)
    "avg_psnr",
    "avg_ssim",
]

# ── Load removal pipeline (Flux2KleinPipeline) ───────────────────────────────
# Models are loaded once at import time; skip with SKIP_REMOVAL=1 for dry-runs.
if os.environ.get("SKIP_REMOVAL") == "1":
    def remove_watermark(image_path, out_dir="recon", scale=0.25):
        raise NotImplementedError("SKIP_REMOVAL mode — removal disabled")
else:
    from compat_flux import remove_watermark

# ── Metrics ───────────────────────────────────────────────────────────────────
from metric import psnr as _psnr, ssim as _ssim


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iter_method_dirs(root):
    """Yield (method_name, sorted list of image Paths) for each sub-dir of root."""
    root = Path(root)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        images = sorted(p for p in d.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if images:
            yield d.name, images


def _to_tensor(pil_img):
    """PIL RGB → (1, 3, H, W) float32 in [0, 1]."""
    return TF.to_tensor(pil_img.convert("RGB")).unsqueeze(0)


def _load_metric_pair(orig_path, recon_path):
    """
    Load both images as tensors, resize the reconstruction to the original
    dimensions if they differ (PSNR/SSIM require same spatial size).
    Returns (orig_tensor, recon_tensor) each (1, 3, H, W) in [0, 1].
    """
    orig  = _to_tensor(Image.open(orig_path))
    recon = _to_tensor(Image.open(recon_path))
    _, _, H, W = orig.shape
    _, _, Hr, Wr = recon.shape
    if H != Hr or W != Wr:
        import torch.nn.functional as F
        recon = F.interpolate(recon, size=(H, W), mode="bilinear",
                              align_corners=False, antialias=True)
    return orig, recon


def _get_detected(result: dict):
    """
    Extract the 'detected' boolean from a verify_watermark return dict.
    Returns None for methods that do not report a detection decision
    (e.g. watermark_anything, which only returns decoded bits).
    """
    if "detected" in result:
        return bool(result["detected"])
    # watermark_anything returns message/message_list — no detection threshold
    return None


def _load_method(name):
    """Import a watermark method by its package name.  Returns None on failure."""
    try:
        return importlib.import_module(name)
    except Exception as e:
        print(f"[{name}] Cannot import: {e}")
        return None


# ── Accuracy accumulator ─────────────────────────────────────────────────────

class _MethodStats:
    def __init__(self):
        self.total = 0
        self.wm_detected_count    = 0   # detected=True before attack
        self.atk_detected_count   = 0   # detected=True after attack
        self.wm_with_decision     = 0   # rows where detected is not None
        self.atk_with_decision    = 0
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.metric_count = 0

    def update(self, wm_detected, attacked_detected, psnr_val, ssim_val):
        self.total += 1
        if wm_detected is not None:
            self.wm_with_decision += 1
            self.wm_detected_count += int(wm_detected)
        if attacked_detected is not None:
            self.atk_with_decision += 1
            self.atk_detected_count += int(attacked_detected)
        if psnr_val != "":
            self.psnr_sum  += float(psnr_val)
            self.ssim_sum  += float(ssim_val)
            self.metric_count += 1

    def summary(self, method_name):
        wm_acc  = (self.wm_detected_count  / self.wm_with_decision
                   if self.wm_with_decision  else None)
        atk_acc = (self.atk_detected_count / self.atk_with_decision
                   if self.atk_with_decision else None)
        avg_psnr = self.psnr_sum / self.metric_count if self.metric_count else None
        avg_ssim = self.ssim_sum / self.metric_count if self.metric_count else None
        return {
            "method":            method_name,
            "total":             self.total,
            "wm_accuracy":       round(wm_acc,  4) if wm_acc  is not None else "",
            "attacked_accuracy": round(atk_acc, 4) if atk_acc is not None else "",
            "avg_psnr":          round(avg_psnr, 4) if avg_psnr is not None else "",
            "avg_ssim":          round(avg_ssim, 4) if avg_ssim is not None else "",
        }


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Per-image CSV — append mode so partial runs are preserved.
    csv_exists = os.path.exists(CSV_PATH)
    csv_file   = open(CSV_PATH, "a", newline="")
    writer     = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if not csv_exists:
        writer.writeheader()

    all_stats = {}   # method_name -> _MethodStats

    try:
        for method_name, images in _iter_method_dirs(TEST_FOLDER):
            mod = _load_method(method_name)
            if mod is None:
                continue

            recon_dir = os.path.join(OUT_DIR, method_name)
            os.makedirs(recon_dir, exist_ok=True)

            stats = _MethodStats()
            all_stats[method_name] = stats

            for img_path in images:
                filename = img_path.name
                row = {f: "" for f in FIELDS}
                row["image"]  = filename
                row["method"] = method_name

                wm_detected = attacked_detected = None

                print(f"\n[{method_name}] {filename}")

                try:
                    # ── 1. Verify watermark on the input image ────────────────
                    verify_wm   = mod.verify_watermark(str(img_path))
                    wm_detected = _get_detected(verify_wm)
                    row["wm_detected"] = wm_detected
                    print(f"  wm verify  : {verify_wm}")

                    # ── 2. Attack: remove watermark ───────────────────────────
                    recon_path = remove_watermark(str(img_path), out_dir=recon_dir)

                    # ── 3. Verify watermark on reconstruction ─────────────────
                    verify_atk      = mod.verify_watermark(recon_path)
                    attacked_detected = _get_detected(verify_atk)
                    row["attacked_detected"] = attacked_detected
                    print(f"  atk verify : {verify_atk}")

                    # ── 4. PSNR / SSIM (watermarked vs reconstruction) ────────
                    wm_t, recon_t  = _load_metric_pair(str(img_path), recon_path)
                    row["psnr"]    = round(_psnr(wm_t, recon_t), 4)
                    row["ssim"]    = round(_ssim(wm_t, recon_t), 4)
                    print(f"  PSNR={row['psnr']} dB  SSIM={row['ssim']}")

                except Exception:
                    err = traceback.format_exc(limit=3)
                    row["error"] = err.replace("\n", " | ")
                    print(f"  ERROR: {err}")

                stats.update(wm_detected, attacked_detected,
                             row["psnr"], row["ssim"])
                writer.writerow(row)
                csv_file.flush()

    finally:
        csv_file.close()

    # ── Per-method accuracy summary ───────────────────────────────────────────
    summary_rows = [s.summary(m) for m, s in all_stats.items()]

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n{'method':<22} {'total':>6} {'wm_acc':>8} {'atk_acc':>8} {'avg_psnr':>10} {'avg_ssim':>9}")
    print("-" * 70)
    for r in summary_rows:
        print(f"{r['method']:<22} {r['total']:>6} {str(r['wm_accuracy']):>8} "
              f"{str(r['attacked_accuracy']):>8} {str(r['avg_psnr']):>10} {str(r['avg_ssim']):>9}")

    print(f"\nPer-image results : {CSV_PATH}")
    print(f"Summary           : {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
