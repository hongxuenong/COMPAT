import importlib
import os

watermarks = ['ssl_watermarking']

image_list = ['sample.jpg']

out_dir = 'output'

for wm in watermarks:
    mod = importlib.import_module(wm)
    wm_dir = os.path.join(out_dir, wm)
    os.makedirs(wm_dir, exist_ok=True)
    for image in image_list:
        if not os.path.exists(image):
            print(f"[{wm}] Skipping {image}: file not found")
            continue
        output_path = os.path.join(wm_dir, os.path.basename(image))
        out = mod.add_watermark(image, output_path=output_path)
        print(f"[{wm}] Watermarked {image} -> {out}")
