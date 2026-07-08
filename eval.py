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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

# Make watermark sub-packages importable as top-level names (dwt_dct, vine, ...)
_WM_DIR = os.path.join(_ROOT, "watermarks")
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
TEST_FOLDER     = "test_folder"      # root containing <wm_method>/<images> sub-dirs
OUT_DIR         = "eval_out"         # root: OUT_DIR/<removal_method>/<wm_method>/scale_*/
CSV_PATH        = "eval_results.csv" # results table
ATTACK_SCALES   = [0.25, 0.5]        # downsample scales to evaluate
REMOVAL_METHODS = ["compat_flux_v2"]    # module names to import remove_watermark from

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── CSV schema ────────────────────────────────────────────────────────────────
FIELDS = [
    "removal_method",   # which removal pipeline was used
    "image",            # filename (not full path)
    "wm_method",        # watermark embedding method name
    "scale",            # downsample scale used for the attack
    "wm_detected",      # watermark present in watermarked image (sanity check)
    "attacked_detected",# watermark present after removal attack
    "psnr",             # dB, watermarked vs reconstruction
    "ssim",             # [−1,1], watermarked vs reconstruction
    "lpips",            # perceptual distance (lower = more similar)
    "error",            # non-empty if any step raised an exception
]

SUMMARY_CSV   = "eval_summary.csv"
SUMMARY_FIELDS = [
    "removal_method",
    "wm_method",
    "scale",
    "total",             # images processed without fatal error
    "wm_accuracy",       # fraction detected before attack  (higher = embedding works)
    "attacked_accuracy", # fraction detected after attack   (lower  = attack works)
    "avg_psnr",
    "avg_ssim",
    "avg_lpips",
]

# ── Metrics ───────────────────────────────────────────────────────────────────
from metric import MetricEvaluator
_metrics = MetricEvaluator()


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


def _load_removal_method(name):
    """Import remove_watermark from a removal method module. Returns None on failure."""
    if os.environ.get("SKIP_REMOVAL") == "1":
        def _stub(image_path, out_dir="recon", scale=0.25):
            raise NotImplementedError("SKIP_REMOVAL mode — removal disabled")
        return _stub
    try:
        return importlib.import_module(name).remove_watermark
    except Exception as e:
        print(f"[removal:{name}] Cannot import: {e}")
        return None


# ── Accuracy accumulator ─────────────────────────────────────────────────────

class _MethodStats:
    def __init__(self):
        self.total = 0
        self.wm_detected_count    = 0   # detected=True before attack
        self.atk_detected_count   = 0   # detected=True after attack
        self.wm_with_decision     = 0   # rows where detected is not None
        self.atk_with_decision    = 0
        self.psnr_sum  = 0.0
        self.ssim_sum  = 0.0
        self.lpips_sum = 0.0
        self.metric_count = 0

    def update(self, wm_detected, attacked_detected, psnr_val, ssim_val, lpips_val):
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
            self.lpips_sum += float(lpips_val)
            self.metric_count += 1

    def summary(self, removal_name, method_name, scale):
        wm_acc  = (self.wm_detected_count  / self.wm_with_decision
                   if self.wm_with_decision  else None)
        atk_acc = (self.atk_detected_count / self.atk_with_decision
                   if self.atk_with_decision else None)
        avg_psnr  = self.psnr_sum  / self.metric_count if self.metric_count else None
        avg_ssim  = self.ssim_sum  / self.metric_count if self.metric_count else None
        avg_lpips = self.lpips_sum / self.metric_count if self.metric_count else None
        return {
            "removal_method":    removal_name,
            "wm_method":         method_name,
            "scale":             scale,
            "total":             self.total,
            "wm_accuracy":       round(wm_acc,  4) if wm_acc  is not None else "",
            "attacked_accuracy": round(atk_acc, 4) if atk_acc is not None else "",
            "avg_psnr":          round(avg_psnr,  4) if avg_psnr  is not None else "",
            "avg_ssim":          round(avg_ssim,  4) if avg_ssim  is not None else "",
            "avg_lpips":         round(avg_lpips, 4) if avg_lpips is not None else "",
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

    all_stats = {}   # (removal_name, wm_method, scale) -> _MethodStats

    try:
        for wm_method, images in _iter_method_dirs(TEST_FOLDER):
            mod = _load_method(wm_method)
            if mod is None:
                continue

            for img_path in images:
                filename = img_path.name
                print(f"\n[{wm_method}] {filename}")

                # Load original tensor once; reused across all removal methods and scales.
                wm_t = _to_tensor(Image.open(img_path))

                # ── 1. Verify watermark on the input image (once per image) ──
                wm_detected = None
                try:
                    verify_wm   = mod.verify_watermark(str(img_path))
                    wm_detected = _get_detected(verify_wm)
                    print(f"  wm verify  : {verify_wm}")
                except Exception:
                    print(f"  wm verify ERROR: {traceback.format_exc(limit=2)}")

                # ── 2–4. For each removal method × scale: attack + verify + metrics ──
                for removal_name in REMOVAL_METHODS:
                    remove_wm = _load_removal_method(removal_name)
                    if remove_wm is None:
                        continue

                    for scale in ATTACK_SCALES:
                        key = (removal_name, wm_method, scale)
                        if key not in all_stats:
                            all_stats[key] = _MethodStats()
                        stats = all_stats[key]

                        recon_dir = os.path.join(OUT_DIR, removal_name, wm_method, f"scale_{scale}")
                        os.makedirs(recon_dir, exist_ok=True)

                        if os.path.exists(os.path.join(recon_dir, filename)):
                            print(f"  [{removal_name}] scale={scale}  skip (output exists)")
                            continue

                        row = {f: "" for f in FIELDS}
                        row["removal_method"] = removal_name
                        row["image"]          = filename
                        row["wm_method"]      = wm_method
                        row["scale"]          = scale
                        row["wm_detected"]    = wm_detected
                        attacked_detected     = None

                        try:
                            recon_path = remove_wm(str(img_path),
                                                    out_dir=recon_dir,
                                                    scale=scale)

                            # Overlap verify (CPU) with loading the recon tensor (disk I/O).
                            def _load_recon(p):
                                t = _to_tensor(Image.open(p).convert("RGB"))
                                if wm_t.shape[-2:] != t.shape[-2:]:
                                    t = F.interpolate(t, size=wm_t.shape[-2:],
                                                      mode="bilinear", align_corners=False,
                                                      antialias=True)
                                return t

                            with ThreadPoolExecutor(max_workers=2) as tex:
                                fut_verify = tex.submit(mod.verify_watermark, recon_path)
                                fut_recon  = tex.submit(_load_recon, recon_path)
                                verify_atk = fut_verify.result()
                                recon_t    = fut_recon.result()

                            attacked_detected = _get_detected(verify_atk)
                            row["attacked_detected"] = attacked_detected
                            print(f"  [{removal_name}] scale={scale}  atk verify : {verify_atk}")

                            row["psnr"]   = round(_metrics.psnr(wm_t, recon_t), 4)
                            row["ssim"]   = round(_metrics.ssim(wm_t, recon_t), 4)
                            row["lpips"]  = round(_metrics.lpips(wm_t, recon_t), 4)
                            print(f"  [{removal_name}] scale={scale}  PSNR={row['psnr']} dB  "
                                  f"SSIM={row['ssim']}  LPIPS={row['lpips']}")

                        except Exception:
                            err = traceback.format_exc(limit=3)
                            row["error"] = err.replace("\n", " | ")
                            print(f"  [{removal_name}] scale={scale}  ERROR: {err}")

                        stats.update(wm_detected, attacked_detected,
                                     row["psnr"], row["ssim"], row["lpips"])
                        writer.writerow(row)
                        csv_file.flush()

    finally:
        csv_file.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_rows = [s.summary(rm, wm, sc)
                    for (rm, wm, sc), s in sorted(all_stats.items())]

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n{'removal':<20} {'wm_method':<18} {'scale':>6} {'total':>6} "
          f"{'wm_acc':>8} {'atk_acc':>8} {'avg_psnr':>10} {'avg_ssim':>9} {'avg_lpips':>10}")
    print("-" * 100)
    for r in summary_rows:
        print(f"{r['removal_method']:<20} {r['wm_method']:<18} {str(r['scale']):>6} {r['total']:>6} "
              f"{str(r['wm_accuracy']):>8} {str(r['attacked_accuracy']):>8} "
              f"{str(r['avg_psnr']):>10} {str(r['avg_ssim']):>9} {str(r['avg_lpips']):>10}")

    print(f"\nPer-image results : {CSV_PATH}")
    print(f"Summary           : {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
