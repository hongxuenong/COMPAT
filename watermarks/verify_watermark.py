import importlib
import os

watermarks = [
    'dwt_dct',
    'ssl_watermarking',
    'trustmark',
    'watermark_anything',
]

image_list = ['sample.jpg']

for wm in watermarks:
    mod = importlib.import_module(wm)
    for image in image_list:
        if not os.path.exists(image):
            print(f"[{wm}] Skipping {image}: file not found")
            continue
        result = mod.verify_watermark(image)
        print(f"[{wm}] {image}: {result}")
