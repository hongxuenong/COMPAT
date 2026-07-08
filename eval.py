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

    out_dir     = out_dir     or f"eval_{attack}_out"
    csv_path    = csv_path    or f"eval_{attack}_results.csv"
    summary_csv = summary_csv or f"eval_{attack}_summary.csv"

    os.makedirs(out_dir, exist_ok=True)

    remove_watermark = load_remover(attack, skip_removal=skip_removal)

    # Per-image CSV — append mode so partial runs are preserved.
    csv_exists = os.path.exists(csv_path)
    csv_file   = open(csv_path, "a", newline="")
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

    with open(summary_csv, "w", newline="") as f:
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

    print(f"\nPer-image results : {csv_path}")
    print(f"Summary           : {summary_csv}")
    return summary_rows


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_attack_kwargs(args):
    """Build the attack_kwargs dict from the relevant CLI options."""
    kwargs = {}
    if args.scale is not None:
        kwargs["scale"] = args.scale
    if args.steps is not None:
        kwargs["num_inference_steps"] = args.steps
    if args.xy is not None:
        kwargs["xy"] = args.xy
    for item in (args.attack_arg or []):
        if "=" not in item:
            raise ValueError(f"--attack-arg must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        # best-effort type coercion
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        kwargs[k] = v
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Unified watermark-removal evaluation (COMPAT / NFPA / ...).")
    parser.add_argument("--attack", default="compat",
                        help=f"removal attack: {', '.join(list_attacks())} (default: compat)")
    parser.add_argument("--test-folder", default=TEST_FOLDER,
                        help="root with <method>/<images> sub-dirs (default: test_folder)")
    parser.add_argument("--out-dir", default=None,
                        help="output dir for reconstructions (default: eval_<attack>_out)")
    parser.add_argument("--csv", default=None,
                        help="per-image results CSV (default: eval_<attack>_results.csv)")
    parser.add_argument("--summary", default=None,
                        help="per-method summary CSV (default: eval_<attack>_summary.csv)")
    parser.add_argument("--skip-removal", action="store_true",
                        help="dry run: do not load/run any attack model")
    # Attack hyper-parameters (only the relevant ones are forwarded).
    parser.add_argument("--scale", type=float, default=None,
                        help="[compat/compat_sr] downsample factor (default 0.25)")
    parser.add_argument("--steps", type=int, default=None,
                        help="[nfpa] DDIM inversion/denoising steps (default 10)")
    parser.add_argument("--xy", type=int, default=None,
                        help="[nfpa] motion-field warp range (default 40)")
    parser.add_argument("--attack-arg", action="append", default=None,
                        metavar="KEY=VALUE",
                        help="extra kwarg forwarded to the removal function (repeatable)")
    args = parser.parse_args()

    run_evaluation(
        attack=args.attack,
        test_folder=args.test_folder,
        out_dir=args.out_dir,
        csv_path=args.csv,
        summary_csv=args.summary,
        attack_kwargs=_parse_attack_kwargs(args),
        skip_removal=args.skip_removal,
    )


if __name__ == "__main__":
    main()
