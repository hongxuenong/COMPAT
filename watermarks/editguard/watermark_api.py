"""
EditGuard watermark wrapper.

Exposes a uniform interface over the upstream EditGuard inference code
(https://github.com/xuanyuzhang21/EditGuard):

    add_watermark(image_path, output_path=None, message=None, ckpt_path=None) -> output_path
    verify_watermark(image_path, ckpt_path=None, message=None) -> dict

EditGuard embeds a 64-bit copyright message plus a localization watermark into
an image and, on extraction, recovers the bits and predicts a tamper mask.

Import isolation
----------------
EditGuard's upstream source uses *flat* absolute imports (``import options.options``,
``from models import create_model``, ``from data import ...``, ``from utils import util``).
Those top-level names (``options``, ``models``, ``data``, ``utils``) collide with
modules from other watermarking methods loaded in the same process (e.g.
``ssl_watermarking`` also defines a top-level ``utils``).

To prevent clashes we install a ``sys.meta_path`` finder that intercepts imports
whose top-level package is one of EditGuard's own (``options``/``models``/``data``/
``utils``) and loads them from *this* directory, registering them under a private
prefix (``_editguard_lib.*``) while also binding the public name so the upstream
``import models`` statements resolve to our isolated copy. The upstream source
files are left unmodified.
"""

import importlib.machinery
import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_NAME = '_editguard_lib'

# EditGuard's own top-level module names (flat imports in the upstream source).
_EG_TOP = ('options', 'models', 'data', 'utils')

_CONFIG_PATH = os.path.join(_DIR, 'options', 'test_editguard.yml')
_DEFAULT_CKPT = os.path.join(_DIR, 'checkpoints', 'clean.pth')

_MESSAGE_LENGTH = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------- #
# Import isolation
# --------------------------------------------------------------------------- #
class _EditGuardRedirect:
    """
    Intercepts absolute imports whose top-level package is one of EditGuard's
    own modules and loads them from this directory, registering each both under
    the private ``_editguard_lib.X`` name and the public ``X`` name so the
    upstream flat imports keep working while remaining isolated from other
    methods loaded in the same interpreter.
    """

    def _owns(self, name):
        top = name.split('.', 1)[0]
        return top in _EG_TOP

    def find_module(self, name, path=None):
        return self if (self._active and self._owns(name)) else None

    def __init__(self):
        # Only redirect while EditGuard code is being imported/run, so we do not
        # hijack identically-named modules belonging to other methods.
        self._active = False

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]

        reg = _PKG_NAME + '.' + name
        if reg in sys.modules:
            sys.modules[name] = sys.modules[reg]
            return sys.modules[name]

        parts = name.split('.')
        p = os.path.join(_DIR, *parts)
        init = os.path.join(p, '__init__.py')

        if os.path.isdir(p) and os.path.exists(init):
            # package with __init__.py
            spec = importlib.util.spec_from_file_location(
                reg, init, submodule_search_locations=[p]
            )
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = reg
            sys.modules[reg] = mod
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        elif os.path.isdir(p):
            # namespace package (no __init__.py)
            spec = importlib.machinery.ModuleSpec(reg, None, is_package=True)
            spec.submodule_search_locations = [p]
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = reg
            sys.modules[reg] = mod
            sys.modules[name] = mod
        else:
            # plain module file
            spec = importlib.util.spec_from_file_location(reg, p + '.py')
            if spec is None:
                raise ImportError(f'Cannot find {name!r} under {_DIR}')
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = (reg.rsplit('.', 1)[0]
                               if len(parts) > 1 else _PKG_NAME)
            sys.modules[reg] = mod
            sys.modules[name] = mod
            spec.loader.exec_module(mod)

        return sys.modules[name]


_REDIRECT = None


def _install_redirect():
    global _REDIRECT
    if _REDIRECT is None:
        _REDIRECT = _EditGuardRedirect()
        sys.meta_path.insert(0, _REDIRECT)
    _REDIRECT._active = True


# --------------------------------------------------------------------------- #
# Checkpoint resolution
# --------------------------------------------------------------------------- #
def _resolve_ckpt(ckpt_path=None):
    candidate = ckpt_path or os.environ.get('EDITGUARD_CKPT') or _DEFAULT_CKPT
    candidate = os.path.abspath(candidate)
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            "EditGuard checkpoint not found at '{}'.\n"
            "The 'clean.pth' checkpoint is distributed via the Google Drive link "
            "in the EditGuard GitHub README (https://github.com/xuanyuzhang21/EditGuard) "
            "and cannot be auto-downloaded. Download it and either place it at "
            "'{}' or set the EDITGUARD_CKPT environment variable (or pass "
            "ckpt_path=...).".format(candidate, _DEFAULT_CKPT)
        )
    return candidate


# --------------------------------------------------------------------------- #
# Model loading (cached)
# --------------------------------------------------------------------------- #
_MODEL_CACHE = {}


def _load_model(ckpt_path=None):
    ckpt = _resolve_ckpt(ckpt_path)
    if ckpt in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt]

    _install_redirect()

    # These imports are intercepted by _EditGuardRedirect -> _editguard_lib.*
    import options.options as option
    from models import create_model

    # is_train=True mirrors the demo (test_gradio/app.py): it populates the
    # opt['path'] entries the model constructor expects.
    opt = option.parse(_CONFIG_PATH, is_train=True)
    opt['dist'] = False
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True

    model = create_model(opt)
    model.load_test(ckpt)

    _MODEL_CACHE[ckpt] = model
    return model


# --------------------------------------------------------------------------- #
# Message helpers
# --------------------------------------------------------------------------- #
def _default_message():
    # Deterministic default 64-bit message so verify() has an expected value.
    return '1010' * (_MESSAGE_LENGTH // 4)


def _message_to_array(message):
    """Convert a '0'/'1' string of length _MESSAGE_LENGTH into the
    {-0.5, +0.5} float array EditGuard's network expects."""
    if len(message) != _MESSAGE_LENGTH:
        raise ValueError(
            'message must be a {}-bit string of 0/1; got length {}'.format(
                _MESSAGE_LENGTH, len(message))
        )
    bits = np.array([int(c) for c in message], dtype=np.float32)
    return bits - 0.5


def _load_image_data(image_path, message_arr=None):
    """Replicates test_gradio.load_image for a single image read from disk.

    Returns the dict expected by model.feed_data: 'LQ' (localization template,
    solid blue), 'GT' (host image), 'MES' (message)."""
    import cv2
    img = cv2.imread(os.path.abspath(image_path))
    if img is None:
        raise FileNotFoundError('Could not read image: {}'.format(image_path))
    # cv2 reads BGR; load_image expects an RGB uint8 array (it re-orders to BGR).
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    img_GT = image / 255.
    img_GT = img_GT[:, :, [2, 1, 0]]
    img_GT = torch.from_numpy(
        np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
    ).float().unsqueeze(0)
    img_GT = torch.nn.functional.interpolate(
        img_GT, size=(512, 512), mode='nearest', align_corners=None)
    img_GT = img_GT.unsqueeze(0)

    _, T, C, W, H = img_GT.shape
    list_h = []
    # Localization watermark template: a solid blue image (R=0, G=0, B=255).
    template = Image.new('RGB', (W, H), (0, 0, 255))
    result = np.array(template) / 255.
    expanded = np.expand_dims(result, axis=0)
    expanded = np.repeat(expanded, T, axis=0)
    imgs_LQ = torch.from_numpy(np.ascontiguousarray(expanded)).float()
    imgs_LQ = imgs_LQ.permute(0, 3, 1, 2)
    imgs_LQ = torch.nn.functional.interpolate(
        imgs_LQ, size=(W, H), mode='nearest', align_corners=None)
    imgs_LQ = imgs_LQ.unsqueeze(0)
    list_h.append(imgs_LQ)
    list_h = torch.stack(list_h, dim=0)

    return {'LQ': list_h, 'GT': img_GT, 'MES': message_arr}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def add_watermark(image_path, output_path=None, message=None, ckpt_path=None):
    """
    Embed a 64-bit copyright message and a localization watermark into an image
    using EditGuard.

    Args:
        image_path: Path to the input image.
        output_path: Save path. Defaults to <stem>_watermarked<ext>.
        message: 64-char string of '0'/'1'. A deterministic default is used if None.
        ckpt_path: Path to clean.pth. Resolved via arg -> $EDITGUARD_CKPT ->
                   <editguard_dir>/checkpoints/clean.pth.

    Returns:
        output_path (str)
    """
    image_path = os.path.abspath(image_path)
    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = '{}_watermarked{}'.format(stem, ext or '.png')

    if message is None:
        message = _default_message()

    model = _load_model(ckpt_path)
    message_arr = _message_to_array(message)

    val_data = _load_image_data(image_path, message_arr)
    model.feed_data(val_data)
    # feed_data does not propagate 'MES'; image_hiding() reads self.mes directly.
    model.mes = message_arr

    container = model.image_hiding()  # HWC, BGR, uint8 (util.tensor2img output)

    import cv2
    cv2.imwrite(output_path, container)
    return output_path


def verify_watermark(image_path, ckpt_path=None, message=None):
    """
    Recover the embedded message bits and predict a tamper mask from an image.

    Args:
        image_path: Path to the (possibly tampered) watermarked image.
        ckpt_path: Path to clean.pth (see add_watermark for resolution order).
        message: 64-char '0'/'1' string to compare against. Defaults to the same
                 deterministic default used by add_watermark.

    Returns:
        dict with keys:
            - message (str): decoded 64-bit string.
            - bit_accuracy (float): fraction of bits matching the expected message.
            - detected (bool): True if bit_accuracy >= 0.95.
            - mask_path (str): path to the saved predicted tamper mask.
    """
    image_path = os.path.abspath(image_path)
    if message is None:
        message = _default_message()
    expected = [int(c) for c in message]

    model = _load_model(ckpt_path)

    val_data = _load_image_data(image_path, None)
    model.feed_data(val_data)

    # image_recovery returns (mask: HxW float ndarray, remesg: (1, L) tensor in {0,1})
    threshold = 0.2
    mask, remesg = model.image_recovery(threshold)

    rec_bits = remesg.detach().cpu().numpy()[0].astype(int).tolist()
    decoded = ''.join(str(int(b)) for b in rec_bits)

    n = min(len(rec_bits), len(expected))
    if n > 0:
        matches = sum(1 for i in range(n) if int(rec_bits[i]) == int(expected[i]))
        bit_accuracy = matches / n
    else:
        bit_accuracy = 0.0

    # Save the predicted tamper mask next to the image.
    stem, _ = os.path.splitext(image_path)
    mask_path = '{}_mask.png'.format(stem)
    mask_arr = np.asarray(mask)
    if mask_arr.max() <= 1.0:
        mask_img = (mask_arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        mask_img = mask_arr.clip(0, 255).astype(np.uint8)
    Image.fromarray(mask_img).save(mask_path)

    return {
        'message': decoded,
        'bit_accuracy': bit_accuracy,
        'detected': bit_accuracy >= 0.95,
        'mask_path': mask_path,
    }
