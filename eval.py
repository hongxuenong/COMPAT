"""
eval.py — Unified watermark-removal evaluation.

TEST_FOLDER layout (images are already watermarked):

    test_folder/
        dwt_dct/
            img1.jpg
            img2.png
        vine/
            img1.jpg
        ...

The sub-directory name must match a watermark method package under ``watermarks/``.

For every method sub-directory and every image inside it:
  1. Verify watermark                       → wm_detected
  2. Attack: remove watermark               → <out_dir>/<wm_method>/scale_<s>/<file>
  3. Verify watermark on reconstruction     → attacked_detected
  4. Compute PSNR / SSIM / LPIPS            (watermarked vs reconstruction)

Results are written row-by-row to a CSV so partial runs are preserved.

Usage:
    python eval.py --attack compat           # Flux2Klein removal (default)
    python eval.py --attack compat_flux_v2   # Flux2Klein v2
    python eval.py --attack diffusion        # SD3 diffusion removal
    python eval.py --attack compat --scale 0.5
    python eval.py --attack compat --test-folder my_data --out-dir my_out

    # Or from Python:
    from eval import run_evaluation
    run_evaluation(attack="compat")

Set SKIP_REMOVAL=1 to dry-run without loading any attack model.
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
from tqdm import tqdm

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))

_WM_DIR = os.path.join(_ROOT, "watermarks")
if _WM_DIR not in sys.path:
    sys.path.insert(0, _WM_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
# TEST_FOLDER   = "/data/xuenong_hong/dataset/aigc/watermark_benchmark/watermarked"      # root containing <wm_method>/<images> sub-dirs
# TEST_FOLDER   = "/data/xuenong_hong/dataset/aigc/watermark_benchmark/watermarked"      # root containing <wm_method>/<images> sub-dirs
TEST_FOLDER   = "/data/xuenong_hong/dataset/aigc/watermark_benchmark/watermarked"
ATTACK_SCALES = [0.25, 0.5]        # default downsample scales to evaluate

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── Attack registry ───────────────────────────────────────────────────────────
# name -> module name that exposes remove_watermark(image_path, out_dir, **kwargs)
ATTACKS = {
    "compat":          "compat",
    "compat_flux_v2":  "compat_flux_v2",
    "compat_omnigen2": "compat_omnigen2",
    # "diffusion":      "attacks.diffusion_attack.diffusion_attack",
}

_ALIASES = {
    "sd3": "diffusion",
    "omnigen2": "compat_omnigen2",
}


def list_attacks():
    return list(ATTACKS.keys())


def _resolve_attack_name(name):
    key = name.lower()
    key = _ALIASES.get(key, key)
    if key not in ATTACKS:
        raise ValueError(f"Unknown attack {name!r}. Available: {', '.join(ATTACKS)}")
    return key


def load_remover(attack, skip_removal=False, use_sam2=False, use_lbp=False):
    """
    Return the remove_watermark callable for the named attack.

    If the module exposes a class with a remove_watermark method (e.g. COMPAT),
    instantiate it once and return the bound method so the model is loaded only once.
    Otherwise fall back to the module-level remove_watermark function.
    """
    if skip_removal or os.environ.get("SKIP_REMOVAL") == "1":
        def _stub(image_path, out_dir="recon", **kwargs):
            raise NotImplementedError("SKIP_REMOVAL mode — removal disabled")
        return _stub
    key = _resolve_attack_name(attack)
    mod = importlib.import_module(ATTACKS[key])
    for attr in dir(mod):
        cls = getattr(mod, attr)
        if isinstance(cls, type) and hasattr(cls, "remove_watermark"):
            init_kwargs = {}
            import inspect
            sig = inspect.signature(cls.__init__)
            if "use_sam2" in sig.parameters:
                init_kwargs["use_sam2"] = use_sam2
            if "use_lbp" in sig.parameters:
                init_kwargs["use_lbp"] = use_lbp
            return cls(**init_kwargs).remove_watermark
    return mod.remove_watermark


# ── CSV schema ────────────────────────────────────────────────────────────────
FIELDS = [
    "removal_method",
    "image",
    "wm_method",
    "scale",
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
    "scale",
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

# ── Metrics ───────────────────────────────────────────────────────────────────
from metric import MetricEvaluator
_metrics = MetricEvaluator()


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
        print(f"[{name}] Cannot import: {e}")
        return None


# ── Accuracy accumulator ─────────────────────────────────────────────────────

class _MethodStats:
    def __init__(self):
        self.total = self.wm_detected_count = self.atk_detected_count = 0
        self.wm_with_decision = self.atk_with_decision = 0
        self.psnr_sum = self.ssim_sum = self.lpips_sum = self.clip_sum = 0.0
        self.wm_ba_sum = self.atk_ba_sum = 0.0
        self.wm_ba_count = self.atk_ba_count = 0
        self.metric_count = 0

    def update(self, wm_detected, wm_ba, attacked_detected, atk_ba, psnr_val, ssim_val, lpips_val, clip_val):
        self.total += 1
        if wm_detected is not None:
            self.wm_with_decision += 1
            self.wm_detected_count += int(wm_detected)
        if wm_ba != "":
            self.wm_ba_sum += float(wm_ba)
            self.wm_ba_count += 1
        if attacked_detected is not None:
            self.atk_with_decision += 1
            self.atk_detected_count += int(attacked_detected)
        if atk_ba != "":
            self.atk_ba_sum += float(atk_ba)
            self.atk_ba_count += 1
        if psnr_val != "":
            self.psnr_sum  += float(psnr_val)
            self.ssim_sum  += float(ssim_val)
            self.lpips_sum += float(lpips_val)
            self.metric_count += 1
        if clip_val != "":
            self.clip_sum += float(clip_val)

    def summary(self, removal_name, wm_method, scale):
        def _avg(s, n): return round(s / n, 4) if n else ""
        wm_acc  = _avg(self.wm_detected_count,  self.wm_with_decision)
        atk_acc = _avg(self.atk_detected_count, self.atk_with_decision)
        return {
            "removal_method":            removal_name,
            "wm_method":                 wm_method,
            "scale":                     scale,
            "total":                     self.total,
            "wm_accuracy":               wm_acc,
            "avg_wm_bit_accuracy":       _avg(self.wm_ba_sum,  self.wm_ba_count),
            "attacked_accuracy":         atk_acc,
            "avg_attacked_bit_accuracy": _avg(self.atk_ba_sum, self.atk_ba_count),
            "avg_psnr":                  _avg(self.psnr_sum,  self.metric_count),
            "avg_ssim":                  _avg(self.ssim_sum,  self.metric_count),
            "avg_lpips":                 _avg(self.lpips_sum, self.metric_count),
            "avg_clip_score":            _avg(self.clip_sum,  self.metric_count),
        }


# ── Main evaluation entry point ───────────────────────────────────────────────

def run_evaluation(attack="compat", test_folder=None, out_dir=None,
                   csv_path=None, summary_csv=None, attack_kwargs=None,
                   skip_removal=False, use_sam2=False, use_lbp=False,
                   n_samples=None):
    """
    Run the watermark-removal evaluation for a single attack.

    Args:
        attack:        registered attack name (see list_attacks()).
        test_folder:   root with <wm_method>/<images> sub-dirs.
        out_dir:       where reconstructions are written.
        csv_path:      per-image results CSV.
        summary_csv:   per-method summary CSV.
        attack_kwargs: extra kwargs forwarded to remove_watermark()
                       (e.g. {'scale': 0.25, 'strength': 0.75}).
        skip_removal:  dry-run without loading any model.

    Returns:
        list of per-method summary dicts.
    """
    attack       = _resolve_attack_name(attack)
    attack_kwargs = dict(attack_kwargs or {})

    test_folder = test_folder or TEST_FOLDER
    out_dir     = out_dir     or f"out/eval_{attack}_out"
    csv_path    = csv_path    or f"out/eval_{attack}_out/eval_{attack}_results.csv"
    summary_csv = summary_csv or f"out/eval_{attack}_out/eval_{attack}_summary.csv"

    # If caller passed a single scale, use it; otherwise iterate over defaults.
    if "scale" in attack_kwargs:
        scales = [attack_kwargs.pop("scale")]
    else:
        scales = ATTACK_SCALES

    os.makedirs(out_dir, exist_ok=True)
    remove_wm = load_remover(attack, skip_removal=skip_removal,
                             use_sam2=use_sam2, use_lbp=use_lbp)

    csv_exists = os.path.exists(csv_path)

    # Build set of (wm_method, scale, image) rows already in the CSV so we can skip them
    done = set()
    if csv_exists:
        with open(csv_path, newline="") as _f:
            for r in csv.DictReader(_f):
                done.add((r.get("wm_method", ""), r.get("scale", ""), r.get("image", "")))

    csv_file = open(csv_path, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if not csv_exists:
        writer.writeheader()

    all_stats  = {}
    wm_objects = {}

    # Total progress units = images × scales (each (image, scale) pair = 1 unit)
    method_dirs  = [(m, imgs) for m, imgs in _iter_method_dirs(test_folder)]
    total_units  = sum(len(imgs) for _, imgs in method_dirs) * len(scales)

    pbar = tqdm(total=total_units, unit="img", dynamic_ncols=True)
    try:
        for wm_method, images in method_dirs:
            if wm_method not in ['rivaGan']:
                continue
            mod = _load_method(wm_method)
            if mod is None:
                pbar.update(len(images) * len(scales))
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
                pbar.update(len(images) * len(scales))
                continue

            for img_path in (images[:n_samples] if n_samples else images):
                filename = img_path.name
                pbar.set_description(f"{wm_method}/{filename}")

                pending_scales = [s for s in scales
                                  if (wm_method, str(s), filename) not in done]
                skipped = len(scales) - len(pending_scales)
                if skipped:
                    pbar.update(skipped)
                if not pending_scales:
                    continue

                wm_t = _to_tensor(Image.open(img_path))

                wm_detected = None
                wm_bit_accuracy = ""
                try:
                    verify_wm       = wm.verify_watermark(str(img_path))
                    wm_detected     = _get_detected(verify_wm)
                    wm_bit_accuracy = verify_wm.get("bit_accuracy", "") if isinstance(verify_wm, dict) else ""
                except Exception:
                    tqdm.write(f"wm verify ERROR [{wm_method}] {filename}: "
                               f"{traceback.format_exc(limit=1).strip()}")

                for scale in pending_scales:
                    key = (wm_method, scale)
                    if key not in all_stats:
                        all_stats[key] = _MethodStats()
                    stats = all_stats[key]

                    recon_dir  = os.path.join(out_dir, f"scale_{scale}", wm_method)
                    recon_path = os.path.join(recon_dir, filename)
                    os.makedirs(recon_dir, exist_ok=True)

                    row = {f: "" for f in FIELDS}
                    row["removal_method"]  = attack
                    row["image"]           = filename
                    row["wm_method"]       = wm_method
                    row["scale"]           = scale
                    row["wm_detected"]     = wm_detected
                    row["wm_bit_accuracy"] = wm_bit_accuracy
                    attacked_detected      = None
                    atk_bit_accuracy       = ""

                    try:
                        if not os.path.exists(recon_path):
                            recon_path = remove_wm(str(img_path),
                                                   out_dir=recon_dir,
                                                   scale=scale,
                                                   **attack_kwargs)

                        def _load_recon(p):
                            t = _to_tensor(Image.open(p).convert("RGB"))
                            if wm_t.shape[-2:] != t.shape[-2:]:
                                t = F.interpolate(t, size=wm_t.shape[-2:],
                                                  mode="bilinear", align_corners=False,
                                                  antialias=True)
                            return t

                        with ThreadPoolExecutor(max_workers=2) as tex:
                            fut_verify = tex.submit(wm.verify_watermark, recon_path)
                            fut_recon  = tex.submit(_load_recon, recon_path)
                            verify_atk = fut_verify.result()
                            recon_t    = fut_recon.result()

                        attacked_detected             = _get_detected(verify_atk)
                        atk_bit_accuracy              = verify_atk.get("bit_accuracy", "") if isinstance(verify_atk, dict) else ""
                        row["attacked_detected"]      = attacked_detected
                        row["attacked_bit_accuracy"]  = atk_bit_accuracy

                        row["psnr"]       = round(_metrics.psnr(wm_t, recon_t), 4)
                        row["ssim"]       = round(_metrics.ssim(wm_t, recon_t), 4)
                        row["lpips"]      = round(_metrics.lpips(wm_t, recon_t), 4)
                        row["clip_score"] = round(_metrics.clip_score(wm_t, recon_t), 4)
                        del recon_t

                        pbar.set_postfix(
                            scale=scale,
                            det=str(attacked_detected),
                            psnr=row["psnr"],
                            ssim=row["ssim"],
                        )

                    except Exception:
                        err = traceback.format_exc(limit=3)
                        row["error"] = err.replace("\n", " | ")
                        tqdm.write(f"ERROR [{wm_method}] {filename} scale={scale}: "
                                   f"{err.splitlines()[-1]}")

                    stats.update(wm_detected, wm_bit_accuracy,
                                 attacked_detected, atk_bit_accuracy,
                                 row["psnr"], row["ssim"], row["lpips"], row["clip_score"])
                    writer.writerow(row)
                    csv_file.flush()
                    pbar.update(1)

    finally:
        pbar.close()
        csv_file.close()

    summary_rows = [s.summary(attack, wm, sc)
                    for (wm, sc), s in sorted(all_stats.items())]

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n{'removal':<20} {'wm_method':<18} {'scale':>6} {'total':>6} "
          f"{'wm_acc':>8} {'wm_ba':>8} {'atk_acc':>8} {'atk_ba':>8} "
          f"{'avg_psnr':>10} {'avg_ssim':>9} {'avg_lpips':>10} {'avg_clip':>9}")
    print("-" * 126)
    for r in summary_rows:
        print(f"{r['removal_method']:<20} {r['wm_method']:<18} {str(r['scale']):>6} "
              f"{r['total']:>6} {str(r['wm_accuracy']):>8} {str(r['avg_wm_bit_accuracy']):>8} "
              f"{str(r['attacked_accuracy']):>8} {str(r['avg_attacked_bit_accuracy']):>8} "
              f"{str(r['avg_psnr']):>10} {str(r['avg_ssim']):>9} {str(r['avg_lpips']):>10} "
              f"{str(r['avg_clip_score']):>9}")

    print(f"\nPer-image results : {csv_path}")
    print(f"Summary           : {summary_csv}")
    return summary_rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified watermark-removal evaluation.")
    parser.add_argument("--attack", default="compat",
                        help=f"removal attack: {', '.join(list_attacks())} (default: compat)")
    parser.add_argument("--test-folder", default=None,
                        help=f"root with <method>/<images> sub-dirs (default: {TEST_FOLDER})")
    parser.add_argument("--out-dir", default=None,
                        help="output dir for reconstructions (default: eval_<attack>_out)")
    parser.add_argument("--csv", default=None,
                        help="per-image results CSV (default: eval_<attack>_results.csv)")
    parser.add_argument("--summary", default=None,
                        help="summary CSV (default: eval_<attack>_summary.csv)")
    parser.add_argument("--skip-removal", action="store_true",
                        help="dry run: do not load/run any attack model")
    parser.add_argument("--use-sam2", action="store_true",
                        help="enable SAM2 segmentation features in COMPAT")
    parser.add_argument("--use-lbp",  action="store_true",
                        help="enable LBP texture features in COMPAT")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="max images to evaluate per watermark folder (default: all)")
    parser.add_argument("--scale", type=float, default=None,
                        help="single downsample scale (default: use ATTACK_SCALES list)")
    parser.add_argument("--steps", type=int, default=None,
                        help="num_inference_steps forwarded to the attack")
    parser.add_argument("--strength", type=float, default=None,
                        help="img2img strength forwarded to the attack (0–1)")
    parser.add_argument("--attack-arg", action="append", default=None,
                        metavar="KEY=VALUE",
                        help="extra kwarg forwarded to remove_watermark() (repeatable)")
    args = parser.parse_args()

    attack_kwargs = {}
    if args.scale    is not None: attack_kwargs["scale"]                = args.scale
    if args.steps    is not None: attack_kwargs["num_inference_steps"]  = args.steps
    if args.strength is not None: attack_kwargs["strength"]             = args.strength
    for item in (args.attack_arg or []):
        if "=" not in item:
            raise ValueError(f"--attack-arg must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:    v = int(v)
        except ValueError:
            try: v = float(v)
            except ValueError: pass
        attack_kwargs[k] = v

    run_evaluation(
        attack=args.attack,
        test_folder=args.test_folder,
        out_dir=args.out_dir,
        csv_path=args.csv,
        summary_csv=args.summary,
        attack_kwargs=attack_kwargs,
        skip_removal=args.skip_removal,
        use_sam2=args.use_sam2,
        use_lbp=args.use_lbp,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
