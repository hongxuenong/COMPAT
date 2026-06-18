import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder

_DEFAULT_WATERMARK = b'watermark'
_DEFAULT_METHOD = 'dwtDct'


def add_watermark(image_path, output_path=None, watermark=_DEFAULT_WATERMARK, method=_DEFAULT_METHOD):
    """
    Embed a watermark into an image using DWT-DCT frequency-domain encoding.

    Args:
        image_path: Path to the input image (must be at least 256x256).
        output_path: Save path. Defaults to <stem>_watermarked<ext>.
        watermark: Bytes payload to embed (default b'watermark').
        method: 'dwtDct' (fast) or 'dwtDctSvd' (more robust).

    Returns:
        output_path (str)
    """
    image_path = os.path.abspath(image_path)
    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = f"{stem}_watermarked{ext or '.png'}"

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', watermark)
    wm_img = encoder.encode(img, method)
    cv2.imwrite(output_path, wm_img)
    return output_path


def verify_watermark(image_path, watermark=_DEFAULT_WATERMARK, method=_DEFAULT_METHOD):
    """
    Decode and verify a DWT-DCT watermark from an image.

    Returns:
        dict with keys:
            - detected (bool): True if decoded bytes match the expected watermark.
            - message (bytes): Raw decoded payload.
    """
    img = cv2.imread(os.path.abspath(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    decoder = WatermarkDecoder('bytes', len(watermark))
    decoded = decoder.decode(img, method)
    return {
        'detected': decoded == watermark,
        'message': decoded,
    }
