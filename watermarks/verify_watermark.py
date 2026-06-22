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

for wm in watermarks:
    try:
        mod = importlib.import_module(wm)
    except Exception as e:
        print(f"[{wm}] Skipped (import failed): {e}")
        continue
    for image in image_list:
        if not os.path.exists(image):
            print(f"[{wm}] Skipping {image}: file not found")
            continue
        try:
            result = mod.verify_watermark(image)
            print(f"[{wm}] {image}: {result}")
        except NotImplementedError as e:
            print(f"[{wm}] Skipped: {e}")
        except Exception as e:
            print(f"[{wm}] Error on {image}: {e}")
