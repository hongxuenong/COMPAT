import importlib
import os

watermarks = [
    'dwt_dct',
    'ssl_watermarking',
    'trustmark',
    'watermark_anything',
    'vine',
    'editguard',
    'omniguard',
]

image_list = ['sample.jpg']

out_dir = 'output'

for wm in watermarks:
    try:
        mod = importlib.import_module(wm)
    except Exception as e:
        print(f"[{wm}] Skipped (import failed): {e}")
        continue
    wm_dir = os.path.join(out_dir, wm)
    os.makedirs(wm_dir, exist_ok=True)
    for image in image_list:
        if not os.path.exists(image):
            print(f"[{wm}] Skipping {image}: file not found")
            continue
        try:
            output_path = os.path.join(wm_dir, os.path.basename(image))
            out = mod.add_watermark(image, output_path=output_path)
            print(f"[{wm}] Watermarked {image} -> {out}")
        except NotImplementedError as e:
            print(f"[{wm}] Skipped: {e}")
        except Exception as e:
            print(f"[{wm}] Error on {image}: {e}")
