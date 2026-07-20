"""
eval.py — Unified watermark-removal evaluation.

Runs the same evaluation pipeline against any registered removal *attack*
(COMPAT / NFPA / ...) through a single function, ``run_evaluation(attack=...)``.

TEST_FOLDER layout (images are already watermarked):

    test_folder/
        dwt_dct/
            img1.jpg
            img2.png
        vine/
            img1.jpg
        ...

The sub-directory name must match a watermark method package under ``watermarks/``
(e.g. ``dwt_dct``, ``ssl_watermarking``, ``trustmark``, ``watermark_anything``,
``vine``, ``editguard``, ``omniguard``).

For every method sub-directory and every image inside it:
  1. Verify watermark                       → wm_detected
  2. Attack: remove watermark               → <OUT_DIR>/<method>/<filename>
  3. Verify watermark on reconstruction      → attacked_detected
  4. Compute PSNR / SSIM                     (watermarked vs reconstruction)

Results are written row-by-row to a CSV so partial runs are preserved, plus a
per-method accuracy summary.

Usage:
    python eval.py --attack compat           # Flux2Klein removal (default)
    python eval.py --attack nfpa             # Next-Frame Prediction attack
    python eval.py --attack compat_sr        # Swin2SR + FluxVAE removal
    python eval.py --attack nfpa --steps 10 --xy 40
    python eval.py --attack compat --test-folder my_data --out-dir my_out

    # Or from Python:
    from eval import run_evaluation
    run_evaluation(attack="nfpa")

Set SKIP_REMOVAL=1 (or --skip-removal) to dry-run without loading any attack model.
"""

import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import importlib
import sys
import traceback
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from metric import MetricEvaluator
_metrics = MetricEvaluator()

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

# Make watermark sub-packages importable as top-level names (dwt_dct, vine, ...)
_WM_DIR = os.path.join(_ROOT, "watermarks")
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)

# Make attack packages importable (e.g. `import nfpa`) and the root modules
# (compat_flux, compat) importable.
_ATK_DIR = os.path.join(_ROOT, "attacks")
for _p in (_ATK_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Attack registry ───────────────────────────────────────────────────────────
# name -> (module_name, attribute) for the removal function.  Modules are imported
# lazily so only the selected attack's heavy model is loaded.
#
# Every removal function has the signature
#     remove_watermark(image_path, out_dir="recon", **attack_kwargs) -> out_path
ATTACKS = {
    "compat":    ("compat_flux", "remove_watermark"),  # Flux2Klein diffusion removal
    "nfpa":      ("nfpa",        "remove_watermark"),  # Next-Frame Prediction attack
    "compat_sr": ("compat",      "remove_watermark"),  # Swin2SR + FluxVAE (no diffusion)
}

# Convenient aliases.
_ALIASES = {
    "flux": "compat",
    "compat_flux": "compat",
    "nfp": "nfpa",
}


def list_attacks():
    """Return the list of registered attack names."""
    return list(ATTACKS.keys())


def _resolve_attack_name(name):
    key = name.lower()
    key = _ALIASES.get(key, key)
    if key not in ATTACKS:
        raise ValueError(
            f"Unknown attack {name!r}. Available: {', '.join(ATTACKS)}"
        )
    return key


def load_remover(attack, skip_removal=False):
    """
    Return the ``remove_watermark`` callable for the named attack.

    With ``skip_removal=True`` (or env SKIP_REMOVAL=1) returns a stub that raises
    NotImplementedError, so the rest of the pipeline can be exercised without
    loading any heavy model.
    """
    if skip_removal or os.environ.get("SKIP_REMOVAL") == "1":
        def _stub(image_path, out_dir="recon", **kwargs):
            raise NotImplementedError("SKIP_REMOVAL mode — removal disabled")
        return _stub

    key = _resolve_attack_name(attack)
    mod_name, attr = ATTACKS[key]
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


# ── Config defaults ───────────────────────────────────────────────────────────
TEST_FOLDER = "test_folder"      # root containing <method>/<images> sub-dirs
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── CSV schema ────────────────────────────────────────────────────────────────
FIELDS = [
    "image",            # filename (not full path)
    "method",           # watermark method name
    "attack",           # removal attack name
    "wm_detected",      # watermark present in watermarked image (sanity check)
    "wm_key_accurate",
    "attacked_detected",# watermark present after removal attack
    "attacked_bit_accuracy",
    "psnr",             # dB, watermarked vs reconstruction
    "ssim",             # [−1,1], watermarked vs reconstruction
    "clip_score",
    "error",            # non-empty if any step raised an exception
]

SUMMARY_FIELDS = [
    "method",
    "attack",
    "total",             # images processed without fatal error
    "wm_accuracy",       # fraction detected before attack  (higher = embedding works)
    "wm_key_accurate",
    "attacked_accuracy", # fraction detected after attack   (lower  = attack works)
    "avg_attacked_bit_accuracy",
    "avg_psnr",
    "avg_ssim",
    "avg_clip_score",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iter_method_dirs(root):
    """Yield (method_name, sorted list of image Paths) for each sub-dir of root."""
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
    if isinstance(result, dict) and "detected" in result:
        return bool(result["detected"])
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
        self.wm_detected_count    = 0
        self.atk_detected_count   = 0
        self.wm_with_decision     = 0
        self.atk_with_decision    = 0
        self.atk_ba_sum = 0.0
        self.atk_ba_count = 0
        self.psnr_sum = self.ssim_sum = self.clip_sum = 0.0
        self.metric_count = 0

    def update(self, wm_detected, attacked_detected, atk_ba, psnr_val, ssim_val, clip_val):
        self.total += 1
        if wm_detected is not None:
            self.wm_with_decision += 1
            self.wm_detected_count += int(wm_detected)
        if attacked_detected is not None:
            self.atk_with_decision += 1
            self.atk_detected_count += int(attacked_detected)
        if atk_ba != "":
            self.atk_ba_sum += float(atk_ba)
            self.atk_ba_count += 1
        if psnr_val != "":
            self.psnr_sum += float(psnr_val)
            self.ssim_sum += float(ssim_val)
            self.metric_count += 1
        if clip_val != "":
            self.clip_sum += float(clip_val)

    def summary(self, method_name, attack_name):
        def _avg(s, n): return round(s / n, 4) if n else ""
        wm_acc  = (self.wm_detected_count  / self.wm_with_decision
                   if self.wm_with_decision  else None)
        atk_acc = (self.atk_detected_count / self.atk_with_decision
                   if self.atk_with_decision else None)
        return {
            "method":                    method_name,
            "attack":                    attack_name,
            "total":                     self.total,
            "wm_accuracy":               round(wm_acc,  4) if wm_acc  is not None else "",
            "attacked_accuracy":         round(atk_acc, 4) if atk_acc is not None else "",
            "avg_attacked_bit_accuracy": _avg(self.atk_ba_sum, self.atk_ba_count),
            "avg_psnr":                  _avg(self.psnr_sum,  self.metric_count),
            "avg_ssim":                  _avg(self.ssim_sum,  self.metric_count),
            "avg_clip_score":            _avg(self.clip_sum,  self.metric_count),
        }


# ── Main evaluation entry point ───────────────────────────────────────────────

def run_evaluation(attack="compat", test_folder=TEST_FOLDER, out_dir=None,
                   csv_path=None, summary_csv=None, attack_kwargs=None,
                   skip_removal=False):
    """
    Run the watermark-removal evaluation for a single attack method.

    Args:
        attack:        registered attack name (see ``list_attacks()``):
                       'compat' (Flux2Klein), 'nfpa' (Next-Frame Prediction),
                       'compat_sr' (Swin2SR + FluxVAE).
        test_folder:   root containing <method>/<images> sub-directories.
        out_dir:       where reconstructions are written
                       (default ``eval_<attack>_out``).
        csv_path:      per-image results CSV (default ``eval_<attack>_results.csv``).
        summary_csv:   per-method summary CSV (default ``eval_<attack>_summary.csv``).
        attack_kwargs: dict of extra kwargs forwarded to the removal function
                       (e.g. {'scale': 0.25} for compat, {'num_inference_steps': 10,
                       'xy': 40} for nfpa).
        skip_removal:  if True, do not load/run the attack model (dry run).

    Returns:
        list of per-method summary dicts.
    """
    attack = _resolve_attack_name(attack)
    attack_kwargs = dict(attack_kwargs or {})

    out_dir     = out_dir     or f"out/eval_{attack}_out"
    csv_path    = csv_path    or f"out/eval_{attack}_results.csv"
    summary_csv = summary_csv or f"out/eval_{attack}_summary.csv"

    os.makedirs(out_dir, exist_ok=True)

    remove_watermark = load_remover(attack, skip_removal=skip_removal)

    # Per-image CSV — append mode so partial runs are preserved.
    csv_exists = os.path.exists(csv_path)

    # Build set of (method, image) rows already in the CSV so we can skip them
    done = set()
    if csv_exists:
        with open(csv_path, newline="") as _f:
            for r in csv.DictReader(_f):
                done.add((r.get("method", ""), r.get("image", "")))

    csv_file = open(csv_path, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if not csv_exists:
        writer.writeheader()

    all_stats  = {}
    wm_objects = {}

    # Collect all (method, images) upfront so tqdm knows the total
    method_dirs = [(m, imgs) for m, imgs in _iter_method_dirs(test_folder)]
    total_images = sum(len(imgs) for _, imgs in method_dirs)

    pbar = tqdm(total=total_images, unit="img", dynamic_ncols=True)
    try:
        for method_name, images in method_dirs:
            mod = _load_method(method_name)
            if mod is None:
                pbar.update(len(images))
                continue

            if method_name not in wm_objects:
                try:
                    pbar.set_description(f"Loading {method_name}")
                    wm_objects[method_name] = mod.Watermark()
                except Exception:
                    tqdm.write(f"[{method_name}] Cannot init Watermark: "
                               f"{traceback.format_exc(limit=1).strip()}")
                    wm_objects[method_name] = None
            wm = wm_objects[method_name]
            if wm is None:
                pbar.update(len(images))
                continue

            recon_dir = os.path.join(out_dir, method_name)
            os.makedirs(recon_dir, exist_ok=True)

            stats = _MethodStats()
            all_stats[method_name] = stats

            for img_path in images:
                filename = img_path.name
                pbar.set_description(f"{method_name}/{filename}")

                if (method_name, filename) in done:
                    pbar.update(1)
                    continue

                row = {f: "" for f in FIELDS}
                row["image"]  = filename
                row["method"] = method_name
                row["attack"] = attack

                wm_detected = attacked_detected = None
                atk_bit_accuracy = ""
                recon_path = os.path.join(recon_dir, filename)

                try:
                    verify_wm          = wm.verify_watermark(str(img_path))
                    wm_detected        = _get_detected(verify_wm)
                    row["wm_detected"] = wm_detected

                    if not os.path.exists(recon_path):
                        recon_path = remove_watermark(str(img_path), out_dir=recon_dir,
                                                      **attack_kwargs)

                    verify_atk                    = wm.verify_watermark(recon_path)
                    attacked_detected             = _get_detected(verify_atk)
                    atk_bit_accuracy              = verify_atk.get("bit_accuracy", "") if isinstance(verify_atk, dict) else ""
                    row["attacked_detected"]      = attacked_detected
                    row["attacked_bit_accuracy"]  = atk_bit_accuracy

                    wm_t, recon_t     = _load_metric_pair(str(img_path), recon_path)
                    row["psnr"]       = round(_metrics.psnr(wm_t, recon_t), 4)
                    row["ssim"]       = round(_metrics.ssim(wm_t, recon_t), 4)
                    row["clip_score"] = round(_metrics.clip_score(wm_t, recon_t), 4)

                    pbar.set_postfix(
                        det=str(attacked_detected),
                        psnr=row["psnr"],
                        ssim=row["ssim"],
                    )

                except Exception:
                    err = traceback.format_exc(limit=3)
                    row["error"] = err.replace("\n", " | ")
                    tqdm.write(f"ERROR [{method_name}] {filename}: {err.splitlines()[-1]}")

                stats.update(wm_detected, attacked_detected, atk_bit_accuracy,
                             row["psnr"], row["ssim"], row["clip_score"])
                writer.writerow(row)
                csv_file.flush()
                pbar.update(1)

    finally:
        pbar.close()
        csv_file.close()

    # ── Per-method accuracy summary ───────────────────────────────────────────
    summary_rows = [s.summary(m, attack) for m, s in all_stats.items()]

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nAttack : {attack}")
    print(f"{'method':<22} {'total':>6} {'wm_acc':>8} {'atk_acc':>8} "
          f"{'atk_ba':>8} {'avg_psnr':>10} {'avg_ssim':>9} {'avg_clip':>9}")
    print("-" * 86)
    for r in summary_rows:
        print(f"{r['method']:<22} {r['total']:>6} {str(r['wm_accuracy']):>8} "
              f"{str(r['attacked_accuracy']):>8} {str(r['avg_attacked_bit_accuracy']):>8} "
              f"{str(r['avg_psnr']):>10} {str(r['avg_ssim']):>9} {str(r['avg_clip_score']):>9}")

    print(f"\nPer-image results : {csv_path}")
    print(f"Summary           : {summary_csv}")
    return summary_rows

if __name__ == "__main__":
    run_evaluation(attack="nfpa", 
                   test_folder='/data/xuenong_hong/dataset/aigc/watermark_benchmark/watermarked')
