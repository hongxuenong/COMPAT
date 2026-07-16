import csv
import importlib
import os
import sys
import glob
import traceback
from pathlib import Path
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

WATERMARKS = [
    'dwt_dct',
    'ssl_watermarking',
    'trustmark',
    'watermark_anything',
    'vine',
    'editguard',
    'omniguard',
]

image_list = glob.glob('/data/xuenong_hong/dataset/aigc/watermark_benchmark/val2017/*.jpg', recursive=True)

CSV_PATH = "accuracy_results.csv"
FIELDS = ["method", "image", "detected", "bit_accuracy", "error"]

# Load one Watermark instance per method up front
wm_objects = {}
for method in WATERMARKS:
    try:
        mod = importlib.import_module(method)
        wm_objects[method] = mod.Watermark()
    except Exception:
        print(f"[{method}] Skipped (init failed):\n{traceback.format_exc(limit=2)}")
        wm_objects[method] = None

with open(CSV_PATH, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    writer.writeheader()

    # bit_accuracy values per method, for threshold computation
    method_scores = {m: [] for m in WATERMARKS}

    for image in sorted(image_list):
        for method, wm in wm_objects.items():
            row = {"method": method, "image": image, "detected": "", "bit_accuracy": "", "error": ""}
            if wm is None:
                row["error"] = "init failed"
            else:
                try:
                    result = wm.verify_watermark(image)
                    row["detected"] = result.get("detected", "")
                    ba = result.get("bit_accuracy", "")
                    row["bit_accuracy"] = ba
                    if ba != "":
                        method_scores[method].append(float(ba))
                    print(f"[{method}] {Path(image).name}: detected={row['detected']}  bit_accuracy={ba}")
                except Exception:
                    row["error"] = traceback.format_exc(limit=2).replace("\n", " | ")
                    print(f"[{method}] Error on {Path(image).name}: {row['error']}")
            writer.writerow(row)
        csv_file.flush()

print(f"\nResults saved to {CSV_PATH}")

# 1% FPR threshold = 99th percentile of bit_accuracy on clean images
print("\n--- 1% FPR thresholds (99th percentile on clean images) ---")
THRESHOLD_CSV = "fpr1_thresholds.csv"
with open(THRESHOLD_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "n_images", "threshold_1fpr"])
    w.writeheader()
    for method in WATERMARKS:
        scores = method_scores[method]
        if scores:
            threshold = float(np.percentile(scores, 99))
            print(f"  {method:<25} n={len(scores):>4}  threshold={threshold:.4f}")
            w.writerow({"method": method, "n_images": len(scores), "threshold_1fpr": round(threshold, 6)})
        else:
            print(f"  {method:<25} no scores")
            w.writerow({"method": method, "n_images": 0, "threshold_1fpr": ""})

print(f"Thresholds saved to {THRESHOLD_CSV}")
