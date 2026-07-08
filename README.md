# COMPAT — Watermark Robustness Benchmark

COMPAT evaluates how well image watermarks survive **removal attacks**. It pairs a
collection of watermark **embedding/verification** methods with a collection of
watermark **removal attacks**, and measures, for each (method, attack) pair, how
often the watermark still survives and how much the image degrades (PSNR/SSIM).

```
COMPAT/
├── eval.py                 # ⭐ unified evaluation runner (all attacks)
├── eval_nfpa.py            # thin shim: == `python eval.py --attack nfpa`
├── metric.py               # PSNR / SSIM
├── compat_flux.py          # attack: Flux2Klein removal          (name: compat)
├── compat.py               # attack: Swin2SR + FluxVAE removal   (name: compat_sr)
├── model.py                # FluxVAE used by compat.py
├── attacks/
│   ├── README.md           # index of removal attacks
│   └── nfpa/               # attack: Next-Frame Prediction       (name: nfpa)
└── watermarks/
    ├── add_watermark.py    # batch-embed helper
    ├── verify_watermark.py # batch-verify helper
    ├── dwt_dct/  ssl_watermarking/  trustmark/  watermark_anything/
    └── vine/  editguard/  omniguard/
```

Each watermark method package exposes `add_watermark(image_path, output_path=None, ...)`
and `verify_watermark(image_path, ...) -> dict`. Each attack exposes
`remove_watermark(image_path, out_dir="recon", **kwargs) -> output_path`.

---

## 1. Installation

```bash
# Python 3.10 recommended
python -m venv .venv && source .venv/bin/activate

# Core deps
pip install torch torchvision numpy pillow

# Attack deps (diffusion models)
pip install "diffusers>=0.31" transformers accelerate safetensors peft

# Per-watermark-method deps (install the ones you use)
pip install opencv-python PyWavelets   # dwt_dct
# ssl_watermarking / trustmark / watermark_anything / vine / editguard / omniguard
# pull torch + their own requirements — see each folder.
```

A **CUDA GPU is required** for the diffusion-based attacks and for the deep-learning
watermark methods (VINE, EditGuard, OmniGuard, WAM).

### Model weights

| Component | Where weights come from |
|---|---|
| `compat` (Flux2Klein) | Local FLUX.2-klein path. Set `COMPAT_FLUX_MODEL=/path/to/FLUX.2-klein-4B` (edit `compat_flux.py` default otherwise). |
| `compat_sr` | Auto-downloaded from HuggingFace (`caidas/swin2SR-*`, `ai-toolkit/flux2_vae`). |
| `nfpa` | Auto-downloads `stabilityai/stable-diffusion-2-1-base`; override with `NFPA_MODEL=/path`. |
| `dwt_dct`, `ssl_watermarking`, `trustmark`, `watermark_anything`, `vine` | Auto-download / bundled. |
| `editguard` | Manual: download `clean.pth` (Google Drive link in `watermarks/editguard/README_INTEGRATION.md`) → set `EDITGUARD_CKPT`. |
| `omniguard` | Manual: download checkpoints (see `watermarks/omniguard/README_INTEGRATION.md`) → set `OMNIGUARD_CKPT`. |

---

## 2. Prepare the data

The evaluation reads **already-watermarked** images arranged by method:

```
test_folder/
    dwt_dct/
        001.png
        002.png
    vine/
        001.png
        ...
    omniguard/
        001.png
```

> The sub-directory **name must match a watermark package** under `watermarks/`
> (`dwt_dct`, `ssl_watermarking`, `trustmark`, `watermark_anything`, `vine`,
> `editguard`, `omniguard`). The evaluator imports that package to verify the mark.

### Generate watermarked images

Embed your own clean images with each method's API, e.g.:

```python
import sys; sys.path.insert(0, "watermarks")
import dwt_dct, vine            # any method package

clean = "my_images/001.png"
dwt_dct.add_watermark(clean, output_path="test_folder/dwt_dct/001.png")
vine.add_watermark(clean,    output_path="test_folder/vine/001.png")
```

Or use the batch helper `watermarks/add_watermark.py` (edit its `watermarks` and
`image_list` variables, then `python watermarks/add_watermark.py`) and move the
results under `test_folder/<method>/`.

---

## 3. Run the evaluation

One runner, one function, selectable attack:

```bash
python eval.py --attack compat        # Flux2Klein removal (default)
python eval.py --attack nfpa          # Next-Frame Prediction attack
python eval.py --attack compat_sr     # Swin2SR + FluxVAE removal
```

Common options:

```bash
python eval.py --attack nfpa --steps 10 --xy 40        # NFPA hyper-params
python eval.py --attack compat --scale 0.25            # COMPAT downsample factor
python eval.py --attack nfpa --test-folder my_data --out-dir my_out
python eval.py --attack compat --skip-removal          # dry run (no model load)
python eval.py --attack nfpa --attack-arg num_inference_steps=20   # generic kwarg
```

From Python:

```python
from eval import run_evaluation, list_attacks

print(list_attacks())                       # ['compat', 'nfpa', 'compat_sr']
run_evaluation(attack="nfpa")
run_evaluation(attack="compat", attack_kwargs={"scale": 0.25})
```

`CUDA_VISIBLE_DEVICES` defaults to `1` (override in your shell, e.g.
`CUDA_VISIBLE_DEVICES=0 python eval.py --attack nfpa`).

### What it does, per image

1. Verify the watermark on the input → `wm_detected`.
2. Run the chosen attack to reconstruct the image → `eval_<attack>_out/<method>/<file>`.
3. Verify the watermark on the reconstruction → `attacked_detected`.
4. Compute PSNR/SSIM (watermarked vs reconstruction).

---

## 4. Outputs

For attack `<attack>`:

- `eval_<attack>_out/<method>/<filename>` — reconstructed (attacked) images.
- `eval_<attack>_results.csv` — per-image rows:
  `image, method, attack, wm_detected, attacked_detected, psnr, ssim, error`.
- `eval_<attack>_summary.csv` — per-method aggregates:
  `method, attack, total, wm_accuracy, attacked_accuracy, avg_psnr, avg_ssim`.

The per-image CSV is appended row-by-row (and flushed), so an interrupted run keeps
its partial results.

### Reading the numbers

- **`wm_accuracy`** — fraction detected *before* the attack. Should be **high**
  (sanity check that embedding + verification work).
- **`attacked_accuracy`** — fraction still detected *after* the attack. **Lower means
  the attack is more effective** at removing the watermark.
- **`avg_psnr` / `avg_ssim`** — similarity of the attacked image to the watermarked
  input. **Higher means the attack preserved image quality** while removing the mark.
  A strong attack achieves low `attacked_accuracy` *and* high PSNR/SSIM.

> Methods that only decode bits without a detection threshold (e.g.
> `watermark_anything`) report `wm_detected`/`attacked_detected` as blank and are
> excluded from the accuracy columns.

---

## 5. Adding a new attack

1. Create a module exposing
   `remove_watermark(image_path, out_dir="recon", **kwargs) -> output_path`
   (put model-based attacks under `attacks/<name>/`).
2. Register it in `eval.py`:

   ```python
   ATTACKS = {
       ...,
       "myattack": ("my_module_or_package", "remove_watermark"),
   }
   ```

3. Run `python eval.py --attack myattack`. (Modules are imported lazily, so only the
   selected attack's weights are loaded.)

See `attacks/README.md` for the current attack list and `attacks/nfpa/` for a full
example.
