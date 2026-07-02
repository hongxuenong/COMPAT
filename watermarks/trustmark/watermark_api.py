import importlib.util
import os
import sys

from PIL import Image

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.join(_DIR, 'python', 'trustmark')
_PKG_NAME = '_trustmark_pkg'


class _TrustMarkRedirect:
    """
    Intercepts absolute imports of 'trustmark.X' and loads them from the
    library directory, registering them as '_trustmark_pkg.X'.  This breaks
    the collision between our wrapper folder and the library package name.
    """
    def find_module(self, name, path=None):
        return self if name.startswith('trustmark.') else None

    def load_module(self, name):
        suffix = name[len('trustmark.'):]          # e.g. 'model' or 'unet'
        reg = _PKG_NAME + '.' + suffix             # '_trustmark_pkg.model'
        # Check the private-namespaced copy first — it only exists if our
        # redirect successfully loaded it, so it's safe to reuse.  We do NOT
        # check sys.modules[name] here: that entry may be a partial stub from a
        # previous failed exec_module (e.g. missing torchmetrics on first run),
        # which would cause "module has no attribute 'TrustMark_Arch'".
        if reg in sys.modules:
            sys.modules[name] = sys.modules[reg]
            return sys.modules[name]

        parts = suffix.split('.')
        p = os.path.join(_PKG_DIR, *parts)
        init = os.path.join(p, '__init__.py')
        if os.path.isdir(p) and os.path.exists(init):
            spec = importlib.util.spec_from_file_location(
                reg, init, submodule_search_locations=[p]
            )
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
        elif os.path.isdir(p):
            # Directory with no __init__.py — treat as namespace package
            spec = importlib.machinery.ModuleSpec(reg, None, is_package=True)
            spec.submodule_search_locations = [p]
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = reg
            sys.modules[reg] = mod
            sys.modules[name] = mod
        else:
            spec = importlib.util.spec_from_file_location(reg, p + '.py')
            if spec is None:
                raise ImportError(f'Cannot find {name!r} in {_PKG_DIR}')
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


# Register finder before loading the package so intra-library absolute imports work
if not any(isinstance(f, _TrustMarkRedirect) for f in sys.meta_path):
    sys.meta_path.insert(0, _TrustMarkRedirect())

if _PKG_NAME not in sys.modules:
    _pkg_spec = importlib.util.spec_from_file_location(
        _PKG_NAME,
        os.path.join(_PKG_DIR, '__init__.py'),
        submodule_search_locations=[_PKG_DIR],
    )
    _pkg = importlib.util.module_from_spec(_pkg_spec)
    sys.modules[_PKG_NAME] = _pkg
    try:
        _pkg_spec.loader.exec_module(_pkg)
    except Exception:
        sys.modules.pop(_PKG_NAME, None)
        raise

TrustMark = sys.modules[_PKG_NAME].TrustMark

_DEFAULT_SECRET = 'watermark'
_DEFAULT_MODEL = 'Q'


def add_watermark(image_path, output_path=None, secret=_DEFAULT_SECRET, model_type=_DEFAULT_MODEL):
    """
    Embed a text watermark using TrustMark.

    Args:
        image_path: Path to the input image (any resolution).
        output_path: Save path. Defaults to <stem>_watermarked<ext>.
        secret: ASCII string to embed (max ~7 chars for default ECC).
        model_type: 'Q' (balanced), 'P' (quality), 'B' (base), or 'C' (compact).

    Returns:
        output_path (str)
    """
    image_path = os.path.abspath(image_path)
    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = f"{stem}_watermarked{ext or '.png'}"

    tm = TrustMark(verbose=False, model_type=model_type)
    img = Image.open(image_path).convert('RGB')
    wm_img = tm.encode(img, secret, MODE='text')
    wm_img.save(output_path)
    return output_path


def verify_watermark(image_path, model_type=_DEFAULT_MODEL):
    """
    Decode and verify a TrustMark watermark.

    Returns:
        dict with keys:
            - detected (bool): True if a watermark is present.
            - message (str): Decoded secret string.
    """
    tm = TrustMark(verbose=False, model_type=model_type)
    img = Image.open(os.path.abspath(image_path)).convert('RGB')
    secret, present, _ = tm.decode(img, MODE='text')
    return {
        'detected': bool(present),
        'message': secret,
    }
