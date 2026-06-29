"""
Image quality metrics: PSNR and SSIM.

All functions accept either:
  - torch.Tensor  (B, C, H, W) or (C, H, W), float, in [0, 1] or [-1, 1]
  - numpy.ndarray  (H, W, C) or (H, W), uint8 or float

The `data_range` parameter defines the value range of the inputs
(1.0 for [0,1] tensors, 2.0 for [-1,1] tensors, 255 for uint8).
"""

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_float_tensor(x) -> torch.Tensor:
    """Convert numpy array or tensor to a 4-D float32 tensor (B,C,H,W)."""
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        else:
            x = x.astype(np.float32)
        if x.ndim == 2:            # (H,W) → (1,1,H,W)
            x = x[None, None]
        elif x.ndim == 3:          # (H,W,C) → (1,C,H,W)
            x = x.transpose(2, 0, 1)[None]
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x
    raise TypeError(f"Expected ndarray or Tensor, got {type(x)}")


def _gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    """1-D Gaussian → outer product → replicated per channel."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g)                        # (k, k)
    return kernel.expand(channels, 1, kernel_size, kernel_size)


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────

def psnr(img1, img2, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    Args:
        img1, img2  : images to compare (numpy or tensor, same shape).
        data_range  : maximum possible value (1.0 for [0,1], 255 for uint8,
                      2.0 for [-1,1]).

    Returns:
        PSNR in dB (averaged over the batch if B > 1).
        Returns float('inf') when the images are identical.
    """
    t1 = _to_float_tensor(img1)
    t2 = _to_float_tensor(img2)
    mse = ((t1 - t2) ** 2).mean(dim=[1, 2, 3])   # per-image MSE
    psnr_vals = torch.where(
        mse == 0,
        torch.full_like(mse, float('inf')),
        10.0 * torch.log10(data_range ** 2 / mse),
    )
    return psnr_vals.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────

def ssim(
    img1,
    img2,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Structural Similarity Index (SSIM).

    Computes the mean SSIM map over each image, then averages across the batch.

    Args:
        img1, img2   : images to compare (numpy or tensor, same shape).
        data_range   : value range (1.0, 2.0, or 255).
        kernel_size  : Gaussian window size (default 11).
        sigma        : Gaussian standard deviation (default 1.5).
        k1, k2       : stability constants (default 0.01, 0.03).

    Returns:
        Mean SSIM in [−1, 1] (perfect similarity = 1.0).
    """
    t1 = _to_float_tensor(img1)
    t2 = _to_float_tensor(img2)

    channels = t1.shape[1]
    kernel = _gaussian_kernel(kernel_size, sigma, channels).to(t1.device)
    pad = kernel_size // 2

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Compute local statistics with depthwise convolution
    mu1 = F.conv2d(t1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(t2, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(t1 * t1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(t2 * t2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12   = F.conv2d(t1 * t2, kernel, padding=pad, groups=channels) - mu12

    ssim_map = (
        (2 * mu12   + c1) * (2 * sigma12   + c2)
    ) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return ssim_map.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(img1, img2, data_range: float = 1.0) -> dict:
    """
    Compute both PSNR and SSIM in one call.

    Returns:
        {'psnr': float (dB), 'ssim': float}
    """
    return {
        'psnr': psnr(img1, img2, data_range=data_range),
        'ssim': ssim(img1, img2, data_range=data_range),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Identical images → PSNR=inf, SSIM=1
    x = torch.rand(2, 3, 256, 256)
    print('Identical:', evaluate(x, x))

    # Slightly noisy
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    print('Noisy    :', evaluate(x, y))

    # Numpy uint8
    a = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    b = (y[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    print('uint8    :', evaluate(a, b, data_range=255))
