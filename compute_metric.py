"""
Compute PSNR / SSIM / LPIPS for matched image pairs in two directory trees.

Both src and target must share the same subfolder structure and filenames:
    src/    subfolder/image.jpg
    target/ subfolder/image.jpg   ← paired with the above

Per-image results are written to CSV_PATH; average LPIPS is printed at the end.
"""

import csv
from pathlib import Path

from tqdm import tqdm

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from metric import MetricEvaluator
_metrics = MetricEvaluator()

src      = '/data/xuenong_hong/dataset/aigc/watermark_benchmark/watermarked/'
target   = '/data/zilin_wang/watermark_project/COMPAT/results/nfpa'
CSV_PATH = 'compute_metric_results.csv'

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
FIELDS   = ['file', 'psnr', 'ssim', 'lpips', 'clip_score', 'error']


def _load_pair(p1: Path, p2: Path):
    a = TF.to_tensor(Image.open(p1).convert('RGB')).unsqueeze(0)
    b = TF.to_tensor(Image.open(p2).convert('RGB')).unsqueeze(0)
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode='bilinear',
                          align_corners=False, antialias=True)
    return a, b


def main():
    src_root = Path(src)
    tgt_root = Path(target)

    pairs = []
    for tgt_path in sorted(tgt_root.rglob('*')):
        if not tgt_path.is_file() or tgt_path.suffix.lower() not in IMG_EXTS:
            continue
        rel = tgt_path.relative_to(tgt_root)
        src_path = src_root / rel
        if src_path.exists():
            pairs.append((src_path, tgt_path, rel))

    if not pairs:
        print(f"No matched pairs found between:\n  src:    {src_root}\n  target: {tgt_root}")
        return

    folder_stats = {}   # subfolder str -> {psnr, ssim, lpips, clip_score, count}
    total = {'psnr': 0., 'ssim': 0., 'lpips': 0., 'clip_score': 0., 'count': 0}

    print(f"{'file':<50} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} {'CLIP':>7}")
    print('-' * 84)

    with open(CSV_PATH, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()

        for src_path, tgt_path, rel in tqdm(pairs, desc="computing metrics"):
            folder = str(rel.parent)
            row = {'file': str(rel), 'psnr': '', 'ssim': '', 'lpips': '',
                   'clip_score': '', 'error': ''}
            try:
                a, b = _load_pair(src_path, tgt_path)
                p = _metrics.psnr(a, b)
                s = _metrics.ssim(a, b)
                l = _metrics.lpips(a, b)
                c = _metrics.clip_score(a, b)

                row['psnr']       = round(p, 4)
                row['ssim']       = round(s, 4)
                row['lpips']      = round(l, 4)
                row['clip_score'] = round(c, 4)

                # print(f"{str(rel):<50} {p:>8.3f} {s:>7.4f} {l:>7.4f} {c:>7.4f}")

                if folder not in folder_stats:
                    folder_stats[folder] = {'psnr': 0., 'ssim': 0., 'lpips': 0.,
                                            'clip_score': 0., 'count': 0}
                for key, val in (('psnr', p), ('ssim', s), ('lpips', l), ('clip_score', c)):
                    folder_stats[folder][key] += val
                    total[key] += val
                folder_stats[folder]['count'] += 1
                total['count'] += 1

            except Exception as e:
                row['error'] = str(e)
                print(f"{str(rel):<50} ERROR: {e}")

            writer.writerow(row)

    # Per-subfolder averages
    if folder_stats:
        print('\n' + '─' * 84)
        print(f"{'subfolder':<50} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} {'CLIP':>7}  count")
        print('─' * 84)
        for folder, st in sorted(folder_stats.items()):
            n = st['count']
            print(f"{folder:<50} {st['psnr']/n:>8.3f} {st['ssim']/n:>7.4f} "
                  f"{st['lpips']/n:>7.4f} {st['clip_score']/n:>7.4f}  {n}")

    n = total['count']
    if n:
        print('\n' + '=' * 84)
        print(f"Average LPIPS      : {total['lpips']      / n:.4f}  (over {n} images)")
        print(f"Average CLIP score : {total['clip_score'] / n:.4f}")
        print(f"Average PSNR       : {total['psnr']       / n:.3f} dB")
        print(f"Average SSIM       : {total['ssim']       / n:.4f}")
        print(f"\nResults saved to: {CSV_PATH}")


if __name__ == '__main__':
    main()
