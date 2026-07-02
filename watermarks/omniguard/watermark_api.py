"""
Wrapper around OmniGuard (CVPR 2025) for the COMPAT watermark collection.

OmniGuard ("Hybrid Manipulation Localization via Augmented Versatile Deep Image
Watermarking", the successor of EditGuard) embeds a hidden secret *image* into a
cover image with an invertible neural network, and additionally trains a ViT-based
mask extractor that predicts a tamper-localization mask for edited regions.

This module exposes the standard COMPAT interface:
    add_watermark(image_path, output_path=None, message=None, ckpt_path=None) -> output_path
    verify_watermark(image_path, ckpt_path=None, message=None) -> dict

Upstream source files (config.py, hinet.py, invblock.py, unet.py, model_invert.py,
rrdb_denselayer.py, iml_vit_model.py, iml_transforms.py, Quantization.py, and the
modules/ package) are vendored verbatim next to this file and are imported through a
private namespace ('_omniguard_lib') so their flat top-level module names
(config, hinet, unet, modules, ...) do not clash with the other watermarking
methods that share the same Python process.
"""

import importlib.machinery
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_NAME = '_omniguard_lib'

# OmniGuard's own flat top-level module names that live inside _DIR.  Any absolute
# import of one of these (issued *by* OmniGuard's own source while _active is True)
# is redirected to load from _DIR under the private '_omniguard_lib' prefix.
_OWN_TOP_LEVEL = {
    'config', 'hinet', 'invblock', 'unet', 'model_invert', 'rrdb_denselayer',
    'iml_vit_model', 'iml_transforms', 'Quantization', 'modules',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Checkpoint files expected under the resolved checkpoint directory.
_CKPT_FILES = {
    'net': 'model_checkpoint_01500.pt',  # main invertible hide/reveal network
    'encoder': 'encoder_Q.ckpt',         # bit-module encoder (loaded by Model.__init__)
    'decoder': 'decoder_Q.ckpt',         # bit-module decoder (loaded by Model.__init__)
    'mask': 'checkpoint-175.pth',        # iml_vit_model tamper-mask extractor
}

_DOWNLOAD_MSG = (
    "OmniGuard checkpoints were not found.\n"
    "They are distributed on PKU Disk / Google Drive (not auto-downloadable):\n"
    "  PKU Disk:     https://disk.pku.edu.cn/link/AAB048898581E047DE9519CE140F991B3A "
    "(file checkpoint.zip, code 5bvw)\n"
    "  Google Drive: https://drive.google.com/file/d/1khdBDUDIRIhPIKlV0ictcbTdWLh-WFY_/view\n"
    "Unzip 'checkpoint.zip' and place the following files in a 'checkpoint' directory, then\n"
    "point to it via the ckpt_path argument or the OMNIGUARD_CKPT environment variable:\n"
    "  {files}\n"
    "Default search location: {default}"
).format(files=', '.join(_CKPT_FILES.values()),
         default=os.path.join(_DIR, 'checkpoint'))


# --------------------------------------------------------------------------- #
# Import isolation: redirect OmniGuard's flat absolute imports into _omniguard_lib
# --------------------------------------------------------------------------- #
class _OmniGuardRedirect:
    """
    A meta-path finder that intercepts absolute imports whose top-level name is one
    of OmniGuard's own modules and loads them from _DIR under the '_omniguard_lib'
    prefix.  It is only active while OmniGuard's own code is being imported
    (_active flag), so it never hijacks identically-named modules belonging to other
    watermark methods in the shared process.
    """
    _active = False

    def find_module(self, name, path=None):
        if not _OmniGuardRedirect._active:
            return None
        top = name.split('.')[0]
        if top in _OWN_TOP_LEVEL:
            return self
        return None

    def load_module(self, name):
        # Check the private-namespaced copy first: if it's already there it was
        # successfully loaded by *our* redirect, so just rebind the public name.
        # We deliberately do NOT check sys.modules[name] here — that entry may be
        # a partial stub left over from a previous failed import attempt (e.g. a
        # missing dependency that raised mid-way through exec_module), which would
        # cause "cannot import name 'Model'" even though the module exists.
        reg = _PKG_NAME + '.' + name
        if reg in sys.modules:
            sys.modules[name] = sys.modules[reg]
            return sys.modules[name]

        parts = name.split('.')
        p = os.path.join(_DIR, *parts)
        init = os.path.join(p, '__init__.py')
        if os.path.isdir(p):
            # package (e.g. 'modules', 'modules.foo' that is itself a dir)
            if os.path.exists(init):
                spec = importlib.util.spec_from_file_location(
                    reg, init, submodule_search_locations=[p])
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = reg
                sys.modules[reg] = mod
                sys.modules[name] = mod
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    sys.modules.pop(reg, None)
                    sys.modules.pop(name, None)
                    raise
            else:
                spec = importlib.machinery.ModuleSpec(reg, None, is_package=True)
                spec.submodule_search_locations = [p]
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = reg
                sys.modules[reg] = mod
                sys.modules[name] = mod
        else:
            spec = importlib.util.spec_from_file_location(reg, p + '.py')
            if spec is None:
                raise ImportError(f'Cannot find {name!r} in {_DIR}')
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = (_PKG_NAME + '.' + '.'.join(parts[:-1])
                               if len(parts) > 1 else _PKG_NAME)
            sys.modules[reg] = mod
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop(reg, None)
                sys.modules.pop(name, None)
                raise
        return sys.modules[name]


def _ensure_lib_package():
    """Register the private '_omniguard_lib' namespace package once."""
    if _PKG_NAME not in sys.modules:
        spec = importlib.machinery.ModuleSpec(_PKG_NAME, None, is_package=True)
        spec.submodule_search_locations = [_DIR]
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_PKG_NAME] = pkg
    if not any(isinstance(f, _OmniGuardRedirect) for f in sys.meta_path):
        sys.meta_path.insert(0, _OmniGuardRedirect())


class _redirect_active:
    """Context manager enabling the redirect finder only while OmniGuard imports run."""
    def __enter__(self):
        self._prev = _OmniGuardRedirect._active
        _OmniGuardRedirect._active = True
        return self

    def __exit__(self, *exc):
        _OmniGuardRedirect._active = self._prev
        return False


class _chdir:
    """Temporarily change cwd so OmniGuard's relative 'checkpoint/...' paths resolve."""
    def __init__(self, target):
        self._target = target

    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self._target)
        return self

    def __exit__(self, *exc):
        os.chdir(self._old)
        return False


# --------------------------------------------------------------------------- #
# Checkpoint resolution
# --------------------------------------------------------------------------- #
def _resolve_ckpt_dir(ckpt_path=None):
    """
    Resolve the directory that contains OmniGuard's checkpoint files.
    Order: explicit ckpt_path arg -> OMNIGUARD_CKPT env var -> <dir>/checkpoint.
    ckpt_path / OMNIGUARD_CKPT may be a directory or a path to the main .pt file.
    """
    candidates = []
    if ckpt_path:
        candidates.append(ckpt_path)
    env = os.environ.get('OMNIGUARD_CKPT')
    if env:
        candidates.append(env)
    candidates.append(os.path.join(_DIR, 'checkpoint'))

    for cand in candidates:
        cand = os.path.abspath(cand)
        ckpt_dir = cand if os.path.isdir(cand) else os.path.dirname(cand)
        if os.path.exists(os.path.join(ckpt_dir, _CKPT_FILES['net'])):
            return ckpt_dir

    raise FileNotFoundError(_DOWNLOAD_MSG)


# --------------------------------------------------------------------------- #
# Model construction (cached)
# --------------------------------------------------------------------------- #
_NET_CACHE = {}
_MASK_CACHE = {}


def _load_state(net, name):
    """Load the main invertible network checkpoint (mirrors demo.load())."""
    state_dicts = torch.load(name, map_location='cpu')
    network_state_dict = {k: v for k, v in state_dicts['net'].items()
                          if ('tmp_var' not in k) and ('bm' not in k)}
    net.load_state_dict(network_state_dict, strict=False)


def _staging_root(ckpt_dir):
    """
    OmniGuard's Model.__init__ loads 'checkpoint/encoder_Q.ckpt' and
    'checkpoint/decoder_Q.ckpt' via paths relative to cwd.  Return a directory that
    contains a 'checkpoint' entry resolving to ckpt_dir, so we can chdir into it
    regardless of how the user named their checkpoint directory.
    """
    if os.path.basename(os.path.normpath(ckpt_dir)) == 'checkpoint':
        return os.path.dirname(os.path.normpath(ckpt_dir))

    import tempfile
    stage = tempfile.mkdtemp(prefix='omniguard_ckpt_')
    link = os.path.join(stage, 'checkpoint')
    try:
        os.symlink(ckpt_dir, link)
    except (OSError, NotImplementedError):
        # Fall back to a directory of symlinked files if dir symlink is unavailable.
        os.makedirs(link, exist_ok=True)
        for fname in _CKPT_FILES.values():
            src = os.path.join(ckpt_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(link, fname)
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except (OSError, NotImplementedError):
                        import shutil
                        shutil.copy2(src, dst)
    return stage


def _build_net(ckpt_dir):
    if ckpt_dir in _NET_CACHE:
        return _NET_CACHE[ckpt_dir]

    _ensure_lib_package()
    with _redirect_active(), _chdir(_staging_root(ckpt_dir)):
        from model_invert import Model, init_model
        net = Model()
        net = net.to(device)
        init_model(net)
        _load_state(net, os.path.join(ckpt_dir, _CKPT_FILES['net']))
        net.eval()

    _NET_CACHE[ckpt_dir] = net
    return net


def _build_mask_extractor(ckpt_dir):
    if ckpt_dir in _MASK_CACHE:
        return _MASK_CACHE[ckpt_dir]

    _ensure_lib_package()
    with _redirect_active():
        from iml_vit_model import iml_vit_model
        mask_path = os.path.join(ckpt_dir, _CKPT_FILES['mask'])
        if not os.path.exists(mask_path):
            raise FileNotFoundError(_DOWNLOAD_MSG)
        extractor = iml_vit_model()
        extractor.load_state_dict(torch.load(mask_path, map_location='cpu',weights_only=False)['model'],
                                  strict=True)
        extractor = extractor.to(device).eval()

    _MASK_CACHE[ckpt_dir] = extractor
    return extractor


# --------------------------------------------------------------------------- #
# Secret image
# --------------------------------------------------------------------------- #
def _reference_secret(h, w):
    """
    The upstream demo embeds a fixed secret image ('bluesky_white2.png'), which is a
    binary asset not vendored here.  It is a (near-)white reference image used purely
    as the fixed hidden payload, so we reconstruct an all-white secret of the cover
    size.  Recovery quality of this known reference doubles as the detection signal.
    """
    arr = np.ones((h, w, 3), dtype=np.float32)  # white, matching 'bluesky_white2'
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return t


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def add_watermark(image_path, output_path=None, message=None, ckpt_path=None):
    """
    Embed OmniGuard's hidden secret image into a cover image.

    Args:
        image_path: Path to the cover image.
        output_path: Save path. Defaults to <stem>_watermarked<ext>.
        message: Unused for OmniGuard's image-in-image embedding (kept for API
                 compatibility); the hidden payload is OmniGuard's fixed secret image.
        ckpt_path: Optional checkpoint dir or path to the main .pt file.

    Returns:
        output_path (str)
    """
    image_path = os.path.abspath(image_path)
    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = f"{stem}_watermarked{ext or '.png'}"

    ckpt_dir = _resolve_ckpt_dir(ckpt_path)
    net = _build_net(ckpt_dir)

    with _redirect_active():
        import modules.Unet_common as common
    dwt = common.DWT()

    cover_img = Image.open(image_path).convert('RGB')
    cover = np.array(cover_img) / 255.
    cover = torch.from_numpy(cover).permute(2, 0, 1).unsqueeze(0).float().to(device)

    b, _, Wd, Hd = cover.shape
    secret = _reference_secret(Wd, Hd)

    with torch.no_grad():
        cover_input = dwt(cover)
        secret_input = dwt(secret)
        steg_img, _output_z, _out_temp, _secret_temp = net(cover_input, secret_input)

    steg = steg_img.permute(0, 2, 3, 1).cpu().squeeze().numpy()
    steg = np.clip(steg, 0, 1) * 255.0
    Image.fromarray(steg.astype(np.uint8)).save(output_path)
    return output_path


def verify_watermark(image_path, ckpt_path=None, message=None):
    """
    Recover OmniGuard's hidden secret image and predict a tamper-localization mask.

    Args:
        image_path: Path to the (possibly tampered) watermarked image.
        ckpt_path: Optional checkpoint dir or path to the main .pt file.
        message: Unused (kept for API compatibility).

    Returns:
        dict with keys:
            - message (str): bit string derived from the recovered secret (thresholded).
            - bit_accuracy (float): agreement between recovered secret bits and the
              known reference secret, in [0, 1].
            - detected (bool): True if bit_accuracy >= 0.95.
            - mask_path (str): path to the saved predicted tamper mask (PNG).
    """
    image_path = os.path.abspath(image_path)
    ckpt_dir = _resolve_ckpt_dir(ckpt_path)
    net = _build_net(ckpt_dir)
    mask_extractor = _build_mask_extractor(ckpt_dir)

    with _redirect_active():
        import modules.Unet_common as common
        import albumentations as albu
        from albumentations.pytorch import ToTensorV2
    dwt = common.DWT()
    iwt = common.IWT()

    steg_img = Image.open(image_path).convert('RGB')
    steg = np.array(steg_img) / 255.
    steg = torch.from_numpy(steg).permute(2, 0, 1).unsqueeze(0).float().to(device)

    def _downsample_below_1024(t):
        _, _, H, W = t.shape
        while W > 1024 or H > 1024:
            t = F.interpolate(t, scale_factor=0.5, mode='bilinear', align_corners=False)
            _, _, H, W = t.shape
        return t

    with torch.no_grad():
        steg_input = dwt(steg)
        image_fuse = steg

        output_image = net(steg_input.to(device), rev=True)
        secret_rev = output_image.narrow(1, 0, 12)
        secret_rev = iwt(secret_rev).to(device)

        artifact = _downsample_below_1024(secret_rev.to(device))
        fuse = _downsample_below_1024(image_fuse.to(device))
        b, _, Wd, Hd = artifact.shape

        # Bit signal: compare the recovered secret against the known white reference.
        ref = _reference_secret(Wd, Hd)
        rec_bits = (artifact.clamp(0, 1) >= 0.5)
        ref_bits = (ref >= 0.5)
        bit_accuracy = (rec_bits == ref_bits).float().mean().item()
        bits = rec_bits.flatten().cpu().numpy().astype(np.uint8)
        # Keep the message string bounded so it stays usable.
        msg_str = ''.join(str(int(x)) for x in bits[:256])

        # Tamper mask prediction via the ViT mask extractor.
        outputsize = 1024
        ablu_my = albu.Compose([
            albu.PadIfNeeded(min_height=outputsize, min_width=outputsize,
                             border_mode=0, fill=0, position='top_left', fill_mask=0),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            albu.Crop(0, 0, outputsize, outputsize),
            ToTensorV2(),
        ])

        artifact_np = artifact.permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255
        fuse_np = fuse.permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255

        artifact_t = ablu_my(image=artifact_np)['image'].to(device).unsqueeze(0)
        fuse_t = ablu_my(image=fuse_np)['image'].to(device).unsqueeze(0)

        mask_pred = mask_extractor(artifact_t, fuse_t)
        mask_pred = mask_pred[:, :, 0:Wd, 0:Hd]

    stem, _ = os.path.splitext(image_path)
    mask_path = f"{stem}_mask.png"
    mask_np = mask_pred.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    mask_np = np.clip(mask_np, 0, 1) * 255.0
    Image.fromarray(mask_np.astype(np.uint8)).save(mask_path)

    return {
        'message': msg_str,
        'bit_accuracy': bit_accuracy,
        'detected': bit_accuracy >= 0.95,
        'mask_path': mask_path,
    }
