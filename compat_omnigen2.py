"""
compat_omnigen2.py — OmniGen2 instruction-guided reconstruction attack.

Thin module exposing ``COMPAT_omnigen2`` (defined in ``compat.py``) so the
evaluation harness can select it via the attack registry, e.g.:

    python eval.py --attack compat_omnigen2

The class itself lives in ``compat.py`` and inherits everything from ``COMPAT``
except model loading and ``reconstruct``, which use the OmniGen2 pipeline
(https://github.com/VectorSpaceLab/OmniGen2).
"""

from compat import COMPAT_omnigen2


def remove_watermark(image_path: str, out_dir: str = "recon",
                     degrade_method: str = "scale", **kwargs) -> str:
    """Module-level shim (used when not going through the class-based loader)."""
    return COMPAT_omnigen2().remove_watermark(
        image_path, out_dir=out_dir, degrade_method=degrade_method, **kwargs)
