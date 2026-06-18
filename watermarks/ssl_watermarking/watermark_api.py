import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage

# Allow imports from this directory regardless of working directory
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import data_augmentation
import encode
import decode
import utils
import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default paths (relative to this file)
_DEFAULT_MODEL_PATH = os.path.join(_DIR, "models", "dino_r50_plus.pth")
_DEFAULT_NORMLAYER_PATH = os.path.join(_DIR, "normlayers", "out2048_yfcc_orig.pth")
_DEFAULT_CARRIER_PATH = os.path.join(_DIR, "carriers", "carrier_1_2048.pth")
_DEFAULT_TARGET_PSNR = 42.0
_DEFAULT_TARGET_FPR = 1e-6
_DEFAULT_CARRIERS_DIR = os.path.join(_DIR, "carriers")


def _load_model(model_path=_DEFAULT_MODEL_PATH, normlayer_path=_DEFAULT_NORMLAYER_PATH):
    backbone = utils.build_backbone(path=model_path, name="resnet50")
    normlayer = utils.load_normalization_layer(path=normlayer_path)
    model = utils.NormLayerWrapper(backbone, normlayer)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def _load_carrier_and_angle(model, carrier_path=_DEFAULT_CARRIER_PATH, target_fpr=_DEFAULT_TARGET_FPR):
    D = model(torch.zeros((1, 3, 224, 224)).to(device)).size(-1)
    carrier = torch.load(carrier_path, map_location=device)
    assert D == carrier.shape[1], f"Carrier dimension {carrier.shape[1]} does not match model output {D}"
    carrier = carrier.to(device, non_blocking=True)
    angle = utils.pvalue_angle(dim=D, k=1, proba=target_fpr)
    return carrier, angle


def _load_multibit_carrier(model, num_bits, carriers_dir=_DEFAULT_CARRIERS_DIR):
    """Load or generate a KxD carrier for multi-bit watermarking."""
    D = model(torch.zeros((1, 3, 224, 224)).to(device)).size(-1)
    os.makedirs(carriers_dir, exist_ok=True)
    carrier_path = os.path.join(carriers_dir, f"carrier_{num_bits}_{D}.pth")
    if os.path.exists(carrier_path):
        carrier = torch.load(carrier_path, map_location=device)
    else:
        carrier = utils.generate_carriers(num_bits, D, output_fpath=carrier_path)
    assert carrier.shape == (num_bits, D), f"Carrier shape {carrier.shape} does not match ({num_bits}, {D})"
    return carrier.to(device, non_blocking=True)


def _parse_message(message, num_bits):
    """
    Convert a message to a boolean tensor of shape (1, num_bits).

    Accepts:
        - str of '0'/'1' characters  e.g. "0110100"
        - list/tuple of int (0/1) or bool
        - torch.BoolTensor of shape (num_bits,)
    """
    if isinstance(message, str):
        bits = [c == '1' for c in message]
    elif isinstance(message, torch.Tensor):
        bits = message.bool().tolist()
    else:
        bits = [bool(b) for b in message]

    if len(bits) != num_bits:
        raise ValueError(f"Message length {len(bits)} does not match num_bits={num_bits}")
    return torch.tensor(bits, dtype=torch.bool).unsqueeze(0)  # 1xK


def add_watermark(
    image_path,
    output_path=None,
    model_path=_DEFAULT_MODEL_PATH,
    normlayer_path=_DEFAULT_NORMLAYER_PATH,
    carrier_path=_DEFAULT_CARRIER_PATH,
    target_psnr=_DEFAULT_TARGET_PSNR,
    target_fpr=_DEFAULT_TARGET_FPR,
    epochs=100,
):
    """
    Embed a 0-bit SSL watermark into an image.

    Args:
        image_path: Path to the input image.
        output_path: Where to save the watermarked image. Defaults to
            <stem>_watermarked<ext> next to the original.
        model_path: Path to the backbone checkpoint.
        normlayer_path: Path to the normalization layer checkpoint.
        carrier_path: Path to the carrier tensor.
        target_psnr: PSNR budget in dB (higher = less visible perturbation).
        target_fpr: Target false-positive rate for the detector.
        epochs: Number of optimisation steps.

    Returns:
        output_path (str): Path to the saved watermarked image.
    """
    image_path = os.path.abspath(image_path)
    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = f"{stem}_watermarked{ext if ext else '.png'}"

    model = _load_model(model_path, normlayer_path)
    carrier, angle = _load_carrier_and_angle(model, carrier_path, target_fpr)
    data_aug = data_augmentation.All()

    img = Image.open(image_path).convert("RGB")
    img_tensor = utils_img.default_transform(img).to(device)

    class _SimpleParams:
        verbose = 0
        optimizer = "Adam,lr=0.01"
        scheduler = None
        batch_size = 1
        lambda_w = 1.0
        lambda_i = 1.0

    params = _SimpleParams()
    params.target_psnr = target_psnr
    params.epochs = epochs

    # Build a minimal dataloader from the single image
    dataloader = [(
        [utils_img.default_transform(img).to(device)],
        [0],
    )]

    pt_imgs_out = encode.watermark_0bit(dataloader, carrier, angle, model, data_aug, params)
    wm_img = ToPILImage()(utils_img.unnormalize_img(pt_imgs_out[0]).squeeze(0).clamp(0, 1))
    wm_img.save(output_path)
    return output_path


def add_watermark_multibit(
    image_path,
    message,
    num_bits=None,
    output_path=None,
    model_path=_DEFAULT_MODEL_PATH,
    normlayer_path=_DEFAULT_NORMLAYER_PATH,
    carriers_dir=_DEFAULT_CARRIERS_DIR,
    target_psnr=_DEFAULT_TARGET_PSNR,
    epochs=100,
):
    """
    Embed a multi-bit SSL watermark into an image.

    Args:
        image_path: Path to the input image.
        message: The bits to embed. Accepts a bit-string ("0110..."), a list/tuple
            of int (0/1) or bool, or a torch.BoolTensor of shape (num_bits,).
        num_bits: Number of bits. Inferred from message length when omitted.
        output_path: Where to save the watermarked image. Defaults to
            <stem>_watermarked<ext> next to the original.
        model_path: Path to the backbone checkpoint.
        normlayer_path: Path to the normalization layer checkpoint.
        carriers_dir: Directory for carrier tensors (auto-generated if missing).
        target_psnr: PSNR budget in dB.
        epochs: Number of optimisation steps.

    Returns:
        output_path (str): Path to the saved watermarked image.
    """
    if num_bits is None:
        num_bits = len(message)

    image_path = os.path.abspath(image_path)
    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = f"{stem}_watermarked{ext if ext else '.png'}"

    model = _load_model(model_path, normlayer_path)
    carrier = _load_multibit_carrier(model, num_bits, carriers_dir)
    data_aug = data_augmentation.All()
    msgs = _parse_message(message, num_bits)  # 1xK

    img = Image.open(image_path).convert("RGB")

    class _SimpleParams:
        verbose = 0
        optimizer = "Adam,lr=0.01"
        scheduler = None
        batch_size = 1
        lambda_w = 5e4  # multibit default from main_multibit.py
        lambda_i = 1.0

    params = _SimpleParams()
    params.target_psnr = target_psnr
    params.epochs = epochs

    dataloader = [([utils_img.default_transform(img).to(device)], [0])]

    pt_imgs_out = encode.watermark_multibit(dataloader, msgs, carrier, model, data_aug, params)
    wm_img = ToPILImage()(utils_img.unnormalize_img(pt_imgs_out[0]).squeeze(0).clamp(0, 1))
    wm_img.save(output_path)
    return output_path


def verify_watermark_multibit(
    image_path,
    num_bits,
    model_path=_DEFAULT_MODEL_PATH,
    normlayer_path=_DEFAULT_NORMLAYER_PATH,
    carriers_dir=_DEFAULT_CARRIERS_DIR,
):
    """
    Decode the multi-bit SSL watermark from an image.

    Args:
        image_path: Path to the image to decode.
        num_bits: Number of bits that were embedded (must match what was used at embed time).
        model_path: Path to the backbone checkpoint.
        normlayer_path: Path to the normalization layer checkpoint.
        carriers_dir: Directory containing the carrier tensor.

    Returns:
        dict with keys:
            - message (list[bool]): Decoded bits.
            - message_str (str): Decoded bits as a '0'/'1' string.
    """
    model = _load_model(model_path, normlayer_path)
    carrier = _load_multibit_carrier(model, num_bits, carriers_dir)

    img = Image.open(image_path).convert("RGB")
    results = decode.decode_multibit([img], carrier, model)
    msg = results[0]["msg"].tolist()

    return {
        "message": msg,
        "message_str": "".join("1" if b else "0" for b in msg),
    }


def verify_watermark(
    image_path,
    model_path=_DEFAULT_MODEL_PATH,
    normlayer_path=_DEFAULT_NORMLAYER_PATH,
    carrier_path=_DEFAULT_CARRIER_PATH,
    target_fpr=_DEFAULT_TARGET_FPR,
):
    """
    Detect whether an image carries the 0-bit SSL watermark.

    Args:
        image_path: Path to the image to check.
        model_path: Path to the backbone checkpoint.
        normlayer_path: Path to the normalization layer checkpoint.
        carrier_path: Path to the carrier tensor.
        target_fpr: Target false-positive rate used to set the detection threshold.

    Returns:
        dict with keys:
            - detected (bool): True if the watermark is present.
            - R (float): Acceptance score (positive = inside hypercone).
            - log10_pvalue (float): log10 of the p-value; more negative = more confident.
    """
    model = _load_model(model_path, normlayer_path)
    carrier, angle = _load_carrier_and_angle(model, carrier_path, target_fpr)

    img = Image.open(image_path).convert("RGB")
    results = decode.decode_0bit([img], carrier, angle, model)
    r = results[0]

    # Threshold: log10(target_fpr) — reject if log10_pvalue is below this
    threshold = np.log10(target_fpr)
    detected = r["log10_pvalue"] <= threshold

    return {
        "detected": bool(detected),
        "R": r["R"],
        "log10_pvalue": r["log10_pvalue"],
    }
