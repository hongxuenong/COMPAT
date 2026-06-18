import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys
import urllib.request

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_DIR, 'watermark_anything')   # library subpackage dir
_PKG_NAME = '_wam_lib'

_DEFAULT_CKPT = os.path.join(_DIR, 'checkpoints', 'wam_mit.pth')
_DEFAULT_CONFIG = os.path.join(_DIR, 'checkpoints', 'params.json')
_CKPT_URL = 'https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class _WamRedirect:
    """
    Intercepts absolute imports of 'watermark_anything.X' and loads them from
    the library subdirectory, registered as '_wam_lib.X'.
    """
    def find_module(self, name, path=None):
        return self if name.startswith('watermark_anything.') else None

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        suffix = name[len('watermark_anything.'):]
        reg = _PKG_NAME + '.' + suffix
        if reg not in sys.modules:
            parts = suffix.split('.')
            p = os.path.join(_LIB_DIR, *parts)
            init = os.path.join(p, '__init__.py')
            if os.path.isdir(p) and os.path.exists(init):
                spec = importlib.util.spec_from_file_location(
                    reg, init, submodule_search_locations=[p]
                )
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = reg
                sys.modules[reg] = mod
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            elif os.path.isdir(p):
                # Namespace package — no __init__.py
                spec = importlib.machinery.ModuleSpec(reg, None, is_package=True)
                spec.submodule_search_locations = [p]
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = reg
                sys.modules[reg] = mod
                sys.modules[name] = mod
            else:
                spec = importlib.util.spec_from_file_location(reg, p + '.py')
                if spec is None:
                    raise ImportError(f'Cannot find {name!r} in {_LIB_DIR}')
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = (_PKG_NAME + '.' + '.'.join(parts[:-1])
                                   if len(parts) > 1 else _PKG_NAME)
                sys.modules[reg] = mod
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
        else:
            sys.modules[name] = sys.modules[reg]
        return sys.modules[name]


if not any(isinstance(f, _WamRedirect) for f in sys.meta_path):
    sys.meta_path.insert(0, _WamRedirect())

# Load the library as a namespace package (no __init__.py in the library dir)
if _PKG_NAME not in sys.modules:
    _init = os.path.join(_LIB_DIR, '__init__.py')
    if os.path.exists(_init):
        _pkg_spec = importlib.util.spec_from_file_location(
            _PKG_NAME, _init, submodule_search_locations=[_LIB_DIR]
        )
        _pkg = importlib.util.module_from_spec(_pkg_spec)
        sys.modules[_PKG_NAME] = _pkg
        _pkg_spec.loader.exec_module(_pkg)
    else:
        _pkg_spec = importlib.machinery.ModuleSpec(_PKG_NAME, None, is_package=True)
        _pkg_spec.submodule_search_locations = [_LIB_DIR]
        _pkg = importlib.util.module_from_spec(_pkg_spec)
        sys.modules[_PKG_NAME] = _pkg


def _ensure_checkpoint():
    if not os.path.exists(_DEFAULT_CKPT):
        os.makedirs(os.path.dirname(_DEFAULT_CKPT), exist_ok=True)
        print(f'Downloading WAM checkpoint to {_DEFAULT_CKPT}...')
        urllib.request.urlretrieve(_CKPT_URL, _DEFAULT_CKPT)


def _load_model():
    _ensure_checkpoint()

    with open(_DEFAULT_CONFIG) as f:
        params = json.load(f)

    # Resolve relative config paths against the repo root
    for key in ('embedder_config', 'augmentation_config',
                'extractor_config', 'attenuation_config'):
        params[key] = os.path.join(_DIR, params[key])

    args = argparse.Namespace(**params)

    import omegaconf
    # These imports are intercepted by _WamRedirect → _wam_lib.*
    from watermark_anything.models import build_embedder, build_extractor, Wam
    from watermark_anything.augmentation.augmenter import Augmenter
    from watermark_anything.modules.jnd import JND
    from watermark_anything.data.transforms import unnormalize_img, normalize_img

    embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
    augmenter_cfg = omegaconf.OmegaConf.load(args.augmentation_config)
    attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)

    embedder = build_embedder(args.embedder_model,
                              embedder_cfg[args.embedder_model], args.nbits)
    extractor = build_extractor(extractor_cfg.model,
                                extractor_cfg[args.extractor_model],
                                args.img_size, args.nbits)
    augmenter = Augmenter(**augmenter_cfg)

    attenuation = None
    if getattr(args, 'attenuation', None):
        attenuation = JND(
            **attenuation_cfg[args.attenuation],
            preprocess=unnormalize_img,   # ImageNet-normalized → [0,1] before JND
            postprocess=normalize_img,    # [0,1] → ImageNet-normalized after JND
        )

    wam = Wam(embedder, extractor, augmenter,
              attenuation=attenuation,
              scaling_w=args.scaling_w,
              scaling_i=args.scaling_i).to(device).eval()

    ckpt = torch.load(_DEFAULT_CKPT, map_location='cpu')
    wam.load_state_dict(ckpt)
    wam.to(device)
    return wam


_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD  = [0.229, 0.224, 0.225]

_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGE_MEAN, std=_IMAGE_STD),
])

_UNNORMALIZE = transforms.Normalize(
    mean=[-m / s for m, s in zip(_IMAGE_MEAN, _IMAGE_STD)],
    std=[1 / s for s in _IMAGE_STD],
)


def _to_tensor(img):
    return _NORMALIZE(img).unsqueeze(0).to(device)


def _to_pil(tensor):
    t = _UNNORMALIZE(tensor.squeeze(0).cpu()).clamp(0, 1)
    return transforms.ToPILImage()(t)


def _msg2str(bits):
    return ''.join('1' if b else '0' for b in bits)


def _str2msg(s):
    return [c == '1' for c in s]


def add_watermark(image_path, output_path=None, message=None):
    """
    Embed a 32-bit watermark into an image using Watermark Anything (WAM).

    Args:
        image_path: Path to the input image.
        output_path: Save path. Defaults to <stem>_watermarked<ext>.
        message: 32-char string of '0'/'1'. Random if None.

    Returns:
        output_path (str)
    """
    image_path = os.path.abspath(image_path)
    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = f"{stem}_watermarked{ext or '.png'}"

    wam = _load_model()
    img_pt = _to_tensor(Image.open(image_path).convert('RGB'))

    if message is None:
        wm_msg = wam.get_random_msg(1)
    else:
        wm_msg = torch.tensor(_str2msg(message), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = wam.embed(img_pt, wm_msg)

    _to_pil(outputs['imgs_w']).save(output_path)
    return output_path


def verify_watermark(image_path):
    """
    Detect and decode the WAM watermark from an image.

    Returns:
        dict with keys:
            - message (str): Decoded 32-bit string.
            - message_list (list[bool]): Decoded bits as booleans.
    """
    from watermark_anything.data.metrics import msg_predict_inference

    wam = _load_model()
    img_pt = _to_tensor(Image.open(os.path.abspath(image_path)).convert('RGB'))

    with torch.no_grad():
        preds = wam.detect(img_pt)['preds']

    mask_preds = F.sigmoid(preds[:, 0, :, :])
    bit_preds = preds[:, 1:, :, :]
    pred_msg = msg_predict_inference(bit_preds, mask_preds).cpu().float()[0]

    bits = [bool(b) for b in pred_msg.tolist()]
    return {
        'message': _msg2str(bits),
        'message_list': bits,
    }
