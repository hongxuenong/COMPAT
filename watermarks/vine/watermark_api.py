import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Make the parent of this directory importable so that the upstream
# absolute imports inside the VINE source (``from vine.src... import ...``)
# resolve regardless of the current working directory.
_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from vine.src.vine_turbo import VINE_Turbo
from vine.src.stega_encoder_decoder import CustomConvNeXt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default HuggingFace checkpoints (auto-downloaded on first use).
_DEFAULT_ENCODER = "Shilin-LU/VINE-R-Enc"
_DEFAULT_DECODER = "Shilin-LU/VINE-R-Dec"
_DEFAULT_MESSAGE = "Hello World!"
_SECRET_SIZE = 100  # VINE encodes 100 bits (12 ascii characters + 4 padding bits)

# Cache loaded models so repeated calls do not re-download / re-build them.
_ENCODER_CACHE = {}
_DECODER_CACHE = {}


def crop_to_square(image):
    """Center-crop a PIL image to a square."""
    width, height = image.size
    min_side = min(width, height)
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    return image.crop((left, top, right, bottom))


def _text_to_bits(message):
    """Convert a <=12 character string into a length-100 bit tensor."""
    if len(message) > 12:
        raise ValueError("VINE can only encode 100 bits (12 characters)")
    data = bytearray(message + " " * (12 - len(message)), "utf-8")
    packet_binary = "".join(format(x, "08b") for x in data)
    watermark = [int(x) for x in packet_binary]
    watermark.extend([0, 0, 0, 0])  # pad 96 -> 100 bits
    return torch.tensor(watermark, dtype=torch.float).unsqueeze(0)


def _bits_to_text(bits):
    """Convert the first 96 decoded bits back into a UTF-8 string."""
    chars = []
    for i in range(0, 96, 8):
        byte = bits[i:i + 8]
        value = int("".join(str(int(b)) for b in byte), 2)
        chars.append(value)
    try:
        return bytes(chars).decode("utf-8", errors="replace").rstrip(" \x00")
    except Exception:
        return ""


def _load_encoder(pretrained_model_name=_DEFAULT_ENCODER):
    if pretrained_model_name not in _ENCODER_CACHE:
        enc = VINE_Turbo.from_pretrained(pretrained_model_name)
        enc.to(device)
        enc.eval()
        _ENCODER_CACHE[pretrained_model_name] = enc
    return _ENCODER_CACHE[pretrained_model_name]


def _load_decoder(pretrained_model_name=_DEFAULT_DECODER):
    if pretrained_model_name not in _DECODER_CACHE:
        dec = CustomConvNeXt.from_pretrained(pretrained_model_name)
        dec.to(device)
        dec.eval()
        _DECODER_CACHE[pretrained_model_name] = dec
    return _DECODER_CACHE[pretrained_model_name]


def add_watermark(image_path, output_path=None, message=_DEFAULT_MESSAGE,
                  pretrained_model_name=_DEFAULT_ENCODER):
    """
    Embed a watermark into an image using VINE (generative-prior watermarking).

    Args:
        image_path: Path to the input image.
        output_path: Save path. Defaults to <stem>_watermarked<ext>.
        message: Up to 12 ASCII characters (100-bit payload).
        pretrained_model_name: HuggingFace encoder checkpoint (auto-downloaded).

    Returns:
        output_path (str)
    """
    image_path = os.path.abspath(image_path)
    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = f"{stem}_watermarked{ext or '.png'}"

    encoder = _load_encoder(pretrained_model_name)

    input_image_pil = Image.open(image_path).convert("RGB")
    if input_image_pil.size[0] != input_image_pil.size[1]:
        input_image_pil = crop_to_square(input_image_pil)
    size = input_image_pil.size

    t_val_256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    t_val_orig = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    ])

    resized_img = t_val_256(input_image_pil)            # 256x256, [0,1]
    resized_img = (2.0 * resized_img - 1.0).unsqueeze(0).to(device)
    input_image = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
    input_image = 2.0 * input_image - 1.0

    watermark = _text_to_bits(message).to(device)

    with torch.no_grad():
        encoded_image_256 = encoder(resized_img, watermark)

    # Resolution scaling: apply the 256x256 residual back at the original size.
    residual_256 = encoded_image_256 - resized_img
    residual_orig = t_val_orig(residual_256)
    encoded_image = residual_orig + input_image
    encoded_image = encoded_image * 0.5 + 0.5
    encoded_image = torch.clamp(encoded_image, min=0.0, max=1.0)

    output_pil = transforms.ToPILImage()(encoded_image[0].cpu())
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output_pil.save(output_path)
    return output_path


def verify_watermark(image_path, message=None, pretrained_model_name=_DEFAULT_DECODER):
    """
    Decode and verify the VINE watermark from an image.

    Args:
        image_path: Path to the (possibly edited) watermarked image.
        message: Optional expected message. When given, bit accuracy and a
            detected flag (>= 0.95 bit accuracy) are reported against it; when
            omitted the default message is used as the reference.
        pretrained_model_name: HuggingFace decoder checkpoint (auto-downloaded).

    Returns:
        dict with keys:
            - message (str): Decoded text (first 96 bits as UTF-8).
            - message_bits (str): Decoded 100-bit string.
            - bit_accuracy (float): Accuracy vs. the reference message.
            - detected (bool): True if bit_accuracy >= 0.95.
    """
    decoder = _load_decoder(pretrained_model_name)

    t_val_256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    image = Image.open(os.path.abspath(image_path)).convert("RGB")
    image = t_val_256(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = decoder(image)
    pred = np.array(pred[0].cpu().detach())
    pred = np.round(pred).astype(int)
    pred_bits = pred.tolist()

    reference = _text_to_bits(message if message is not None else _DEFAULT_MESSAGE)
    ref_bits = reference[0].cpu().numpy().astype(int).tolist()

    same = sum(int(x == y) for x, y in zip(ref_bits, pred_bits))
    bit_accuracy = same / _SECRET_SIZE

    return {
        "message": _bits_to_text(pred_bits),
        "message_bits": "".join(str(b) for b in pred_bits),
        "bit_accuracy": bit_accuracy,
        "detected": bool(bit_accuracy >= 0.95),
    }
