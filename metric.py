"""
Image quality metrics: PSNR, SSIM, LPIPS, and CLIP score.

All functions accept either:
  - torch.Tensor  (B, C, H, W) or (C, H, W), float, in [0, 1] or [-1, 1]
  - numpy.ndarray  (H, W, C) or (H, W), uint8 or float

Preferred usage — create one evaluator per process, share it everywhere:
    from metric import MetricEvaluator
    M = MetricEvaluator(device='cuda')
    results = M.evaluate(img1, img2)

Module-level functions (psnr, ssim, lpips, clip_score) are also available and
are backed by a lazily-created default MetricEvaluator.
"""

import numpy as np
import torch
import torch.nn.functional as F
import lpips as _lpips_lib
import open_clip as _open_clip


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
        if x.ndim == 2:
            x = x[None, None]
        elif x.ndim == 3:
            x = x.transpose(2, 0, 1)[None]
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x
    raise TypeError(f"Expected ndarray or Tensor, got {type(x)}")


def _gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g)
    return kernel.expand(channels, 1, kernel_size, kernel_size)


# ViT-B/32 normalisation constants
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def _clip_preprocess_tensor(t: torch.Tensor) -> torch.Tensor:
    """Resize to 224×224 and apply CLIP normalisation. t: (B,3,H,W) in [0,1]."""
    t = F.interpolate(t, size=(224, 224), mode='bicubic', align_corners=False, antialias=True)
    mean = torch.tensor(_CLIP_MEAN, device=t.device).view(1, 3, 1, 1)
    std  = torch.tensor(_CLIP_STD,  device=t.device).view(1, 3, 1, 1)
    return (t - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Pure metric functions (no neural network)
# ─────────────────────────────────────────────────────────────────────────────

def psnr(img1, img2, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB). Returns inf when images are identical."""
    t1 = _to_float_tensor(img1)
    t2 = _to_float_tensor(img2)
    mse = ((t1 - t2) ** 2).mean(dim=[1, 2, 3])
    vals = torch.where(
        mse == 0,
        torch.full_like(mse, float('inf')),
        10.0 * torch.log10(data_range ** 2 / mse),
    )
    return vals.mean().item()


def ssim(
    img1,
    img2,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index. Returns mean SSIM in [−1, 1]."""
    t1 = _to_float_tensor(img1)
    t2 = _to_float_tensor(img2)
    channels = t1.shape[1]
    kernel = _gaussian_kernel(kernel_size, sigma, channels).to(t1.device)
    pad = kernel_size // 2
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu1 = F.conv2d(t1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(t2, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(t1 * t1, kernel, padding=pad, groups=channels) - mu1_sq
    s2 = F.conv2d(t2 * t2, kernel, padding=pad, groups=channels) - mu2_sq
    s12 = F.conv2d(t1 * t2, kernel, padding=pad, groups=channels) - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return ssim_map.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# MetricEvaluator — owns the neural network models
# ─────────────────────────────────────────────────────────────────────────────

class MetricEvaluator:
    """
    Lazy-loads LPIPS (VGG) and CLIP (ViT-B/32) models on first use.
    Create one instance per process and reuse it to avoid redundant loading.

    Args:
        device: torch device string ('cuda', 'cpu', 'cuda:1', …).
                Defaults to 'cuda' if available, else 'cpu'.
        lpips_net: backbone for LPIPS ('vgg', 'alex', 'squeeze').
        clip_model: open_clip model name.
        clip_pretrained: open_clip pretrained weights tag.
    """

    def __init__(
        self,
        device: str = None,
        lpips_net: str = 'vgg',
        clip_model: str = 'ViT-B-32',
        clip_pretrained: str = 'openai',
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self._lpips_net = lpips_net
        self._clip_model_name = clip_model
        self._clip_pretrained = clip_pretrained
        self._lpips_model = (
                _lpips_lib.LPIPS(net=self._lpips_net).to(self.device).eval()
            )
        
        m, _, _ = _open_clip.create_model_and_transforms(
                self._clip_model_name, pretrained=self._clip_pretrained
            )
        self._clip_model = m.to(self.device).eval()

    # ── model accessors (lazy) ────────────────────────────────────────────────

    # ── metric methods ────────────────────────────────────────────────────────

    def psnr(self, img1, img2, data_range: float = 1.0) -> float:
        return psnr(img1, img2, data_range)

    def ssim(self, img1, img2, data_range: float = 1.0, **kwargs) -> float:
        return ssim(img1, img2, data_range, **kwargs)

    def lpips(self, img1, img2, data_range: float = 1.0) -> float:
        t1 = _to_float_tensor(img1).to(self.device)
        t2 = _to_float_tensor(img2).to(self.device)
        if data_range != 2.0:
            t1 = (t1 / data_range) * 2.0 - 1.0
            t2 = (t2 / data_range) * 2.0 - 1.0
        with torch.no_grad():
            dist = self._lpips_model(t1, t2)
        return dist.mean().item()

    def clip_score(self, img1, img2) -> float:
        t1 = _to_float_tensor(img1).to(self.device)
        t2 = _to_float_tensor(img2).to(self.device)
        with torch.no_grad():
            e1 = self._clip_model.encode_image(_clip_preprocess_tensor(t1))
            e2 = self._clip_model.encode_image(_clip_preprocess_tensor(t2))
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = e2 / e2.norm(dim=-1, keepdim=True)
        return (e1 * e2).sum(dim=-1).mean().item()

    def evaluate(self, img1, img2, data_range: float = 1.0) -> dict:
        """Compute all four metrics in one call."""
        return {
            'psnr':       self.psnr(img1, img2, data_range),
            'ssim':       self.ssim(img1, img2, data_range),
            'lpips':      self.lpips(img1, img2, data_range),
            'clip_score': self.clip_score(img1, img2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions (backed by a shared default evaluator)
# ─────────────────────────────────────────────────────────────────────────────

_default_evaluator: 'MetricEvaluator | None' = None


def _get_default() -> MetricEvaluator:
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = MetricEvaluator()
    return _default_evaluator


def lpips(img1, img2, data_range: float = 1.0) -> float:
    return _get_default().lpips(img1, img2, data_range)


def clip_score(img1, img2) -> float:
    return _get_default().clip_score(img1, img2)


def evaluate(img1, img2, data_range: float = 1.0) -> dict:
    return _get_default().evaluate(img1, img2, data_range)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    M = MetricEvaluator()

    x = torch.rand(2, 3, 256, 256)
    print('Identical:', M.evaluate(x, x))

    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    print('Noisy    :', M.evaluate(x, y))

    a = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    b = (y[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    print('uint8    :', M.evaluate(a, b, data_range=255))
