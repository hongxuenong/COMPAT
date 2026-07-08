import os
import sys
import types
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_DIR, 'watermark_anything')

# ── Import isolation ──────────────────────────────────────────────────────────
# When this file is executed as part of the 'watermark_anything' wrapper package,
# sys.modules['watermark_anything'] already points to the partially-initialised
# wrapper.  A plain `from watermark_anything.data.metrics import ...` would either
# (a) look inside the wrapper directory (wrong) or (b) trigger a circular import
# if the wrapper entry is missing.
#
# Solution: replace sys.modules['watermark_anything'] with a bare namespace whose
# __path__ points to the library subdirectory.  Python's import machinery then
# resolves `watermark_anything.data.metrics` → _LIB_DIR/data/metrics.py correctly
# with no __init__.py executed and no circular dependency.  The wrapper module is
# restored immediately after so external callers see add_watermark / verify_watermark.

_wrapper_pkg = sys.modules.get('watermark_anything')

_lib_ns = types.ModuleType('watermark_anything')
_lib_ns.__path__ = [_LIB_DIR]
_lib_ns.__package__ = 'watermark_anything'
sys.modules['watermark_anything'] = _lib_ns

# notebooks/ lives at _DIR/notebooks/
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from watermark_anything.data.metrics import msg_predict_inference  # noqa: E402
from notebooks.inference_utils import (                             # noqa: E402
    load_model_from_checkpoint,
    default_transform,
    create_random_mask,
    unnormalize_img,
    msg2str,
)

# Restore wrapper so `import watermark_anything` still returns add_watermark etc.
if _wrapper_pkg is not None:
    sys.modules['watermark_anything'] = _wrapper_pkg

# ── Device & constants ────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fixed 32-bit message shared by add_watermark (default) and verify_watermark.
_DEFAULT_MESSAGE = torch.tensor([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                   1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

proportion_masked = 0.5

# ── Model (loaded once at import time) ───────────────────────────────────────
_ckpt_dir  = os.path.join(_DIR, 'checkpoints')
_json_path = os.path.join(_ckpt_dir, 'params.json')
_ckpt_path = os.path.join(_ckpt_dir, 'wam_mit.pth')

# Change to _DIR while loading so that any relative config paths inside
# params.json (e.g. augmentation_config: "config/...") resolve correctly.
_old_cwd = os.getcwd()
os.chdir(_DIR)
try:
    wam = load_model_from_checkpoint(_json_path, _ckpt_path).to(device).eval()
finally:
    os.chdir(_old_cwd)

torch.manual_seed(42)


def _load_img(path):
    img = Image.open(os.path.abspath(path)).convert('RGB')
    return default_transform(img).unsqueeze(0).to(device)


# ── Public API ────────────────────────────────────────────────────────────────

def add_watermark(image_path, output_path=None, msg=None):
    """
    Embed a 32-bit watermark into an image using Watermark Anything (WAM).

    Args:
        image_path:  Path to the input image.
        output_path: Save path. Defaults to <stem>_watermarked<ext>.
        msg:         (1, 32) int tensor. Uses _DEFAULT_MESSAGE when None.

    Returns:
        output_path (str)
    """
    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = f'{stem}_watermarked{ext or ".png"}'

    wm_msg  = (_DEFAULT_MESSAGE if msg is None else msg).to(device)
    img_pt  = _load_img(image_path)

    with torch.no_grad():
        outputs = wam.embed(img_pt, wm_msg)

    mask  = create_random_mask(img_pt, num_masks=1, mask_percentage=proportion_masked)
    img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)
    save_image(unnormalize_img(img_w), output_path)
    return output_path


def verify_watermark(image_path, msg=None):
    """
    Detect and decode the WAM watermark from an image.

    Args:
        image_path: Path to the watermarked (or attacked) image.
        msg:        (1, 32) int tensor reference. Uses _DEFAULT_MESSAGE when None.

    Returns:
        dict:
            message (str):        Decoded 32-bit string.
            bit_accuracy (float): Fraction of bits matching the reference.
            detected (bool):      True if bit_accuracy >= 0.95.
    """
    ref_msg = (_DEFAULT_MESSAGE if msg is None else msg).to(device)
    img_w   = _load_img(image_path)

    with torch.no_grad():
        preds = wam.detect(img_w)['preds']

    mask_preds   = torch.sigmoid(preds[:, 0:1, :, :])   # keep (B,1,H,W) for msg_predict_inference
    bit_preds    = preds[:, 1:, :, :]
    pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
    bit_acc      = (pred_message == ref_msg.cpu()).float().mean().item()

    return {
        'message':      msg2str(pred_message[0]),
        'bit_accuracy': bit_acc,
        'detected':     bit_acc >= 0.95,
    }


if __name__ == '__main__':
    _sample = os.path.join(_DIR, '..', 'sample.jpg')
    _out    = add_watermark(_sample)
    print(_out)
    print(verify_watermark(_out))
