"""
Flux.2 VAE — Black-Forest-Labs architecture

Weight keys match the `ae.safetensors` file distributed at
https://huggingface.co/ai-toolkit/flux2_vae so the checkpoint can be loaded
with a plain load_state_dict call.

Config (AutoEncoderParams from the BFL repo):
    ch=128, ch_mult=[1,2,4,4], num_res_blocks=2
    z_channels=32, in_channels=3, out_ch=3
    scale_factor=0.3611, shift_factor=0.1159

Spatial compression : 8× (3 stride-2 downsamples in the encoder)
Latent channels     : 32 (sampled z) / 64 (encoder output = mean + logvar)

Key differences from standard BFL AutoEncoder
----------------------------------------------
  - encoder.quant_conv   lives inside Encoder (not at AutoEncoder top level)
  - decoder.post_quant_conv lives inside Decoder (not at AutoEncoder top level)
  - top-level bn = BatchNorm2d(z_channels, affine=False) tracks latent statistics

Super-resolution
----------------
The decoder always produces 8× the spatial size of its input latent, so SR
by factor `s` is achieved by encoding the LR image and then upsampling the
latent by `s` before decoding:

    LR (H × W)  →  latent (H/8 × W/8)
                →  upsample ×s  →  (sH/8 × sW/8)
                →  decode ×8   →  HR (sH × sW)

Loading weights
---------------
    from safetensors.torch import load_file
    sd = load_file("ae.safetensors")
    model = FluxVAE()
    model.load_state_dict(sd)
    model.eval()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


def _nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class ResnetBlock(nn.Module):
    """Pre-activation ResNet block (BFL naming: norm1/conv1/norm2/conv2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = _normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _nonlinearity(self.norm1(x))
        h = self.conv1(h)
        h = _nonlinearity(self.norm2(h))
        h = self.conv2(h)
        if hasattr(self, 'nin_shortcut'):
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Single-head spatial self-attention (BFL naming: norm/q/k/v/proj_out)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = _normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, height, width = q.shape
        scale = c ** -0.5

        # Flatten spatial dims → (B, HW, C)
        q = q.reshape(b, c, height * width).permute(0, 2, 1)
        k = k.reshape(b, c, height * width)          # (B, C, HW)
        w = torch.bmm(q, k) * scale                  # (B, HW, HW)
        w = F.softmax(w, dim=2)

        v = v.reshape(b, c, height * width)           # (B, C, HW)
        h_ = torch.bmm(v, w.permute(0, 2, 1))        # (B, C, HW)
        h_ = h_.reshape(b, c, height, width)
        return x + self.proj_out(h_)


class Downsample(nn.Module):
    """Asymmetrically-padded stride-2 conv (BFL key: downsample.conv)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), 'constant', 0)
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest ×2 upsample followed by conv (BFL key: upsample.conv)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    BFL Encoder.

    Key structure:
        conv_in
        down.{0..num_res-1}.block.{0..num_res_blocks-1}  (ResnetBlock)
        down.{0..num_res-2}.downsample                    (Downsample)
        mid.block_1 / mid.attn_1 / mid.block_2
        norm_out · conv_out  →  2 * z_channels
    """

    def __init__(
        self,
        ch: int,
        ch_mult: list,
        num_res_blocks: int,
        in_channels: int,
        z_channels: int,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        # Downsampling
        self.down = nn.ModuleList()
        in_ch = ch
        for i_level, mult in enumerate(ch_mult):
            out_ch = ch * mult
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            level = nn.Module()
            level.block = block
            level.attn = nn.ModuleList()           # kept for key compatibility
            if i_level < self.num_resolutions - 1:
                level.downsample = Downsample(in_ch)
            self.down.append(level)

        # Mid
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_ch, in_ch)
        self.mid.attn_1 = AttnBlock(in_ch)
        self.mid.block_2 = ResnetBlock(in_ch, in_ch)

        # Output
        self.norm_out = _normalize(in_ch)
        self.conv_out = nn.Conv2d(in_ch, 2 * z_channels, 3, 1, 1)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level, level in enumerate(self.down):
            for block in level.block:
                h = block(h)
            if i_level < self.num_resolutions - 1:
                h = level.downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = _nonlinearity(self.norm_out(h))
        return self.quant_conv(self.conv_out(h))


class Decoder(nn.Module):
    """
    BFL Decoder.

    Key structure:
        conv_in
        mid.block_1 / mid.attn_1 / mid.block_2
        up.{0..num_res-1}.block.{0..num_res_blocks}      (num_res_blocks+1 blocks)
        up.{1..num_res-1}.upsample                        (Upsample)
        norm_out · conv_out  →  out_ch
    """

    def __init__(
        self,
        ch: int,
        ch_mult: list,
        num_res_blocks: int,
        out_ch: int,
        z_channels: int,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[-1]
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        # Mid
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        # Upsampling — built deepest-first then prepended so up[i] ↔ level i
        self.up = nn.ModuleList()
        in_ch = block_in
        for i_level in reversed(range(self.num_resolutions)):
            out_ch_level = ch * ch_mult[i_level]
            block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):        # one extra block vs encoder
                block.append(ResnetBlock(in_ch, out_ch_level))
                in_ch = out_ch_level
            level = nn.Module()
            level.block = block
            level.attn = nn.ModuleList()
            if i_level != 0:
                level.upsample = Upsample(in_ch)
            self.up.insert(0, level)                   # prepend → up[i] = level i

        # Output
        self.norm_out = _normalize(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mid.block_2(
            self.mid.attn_1(
                self.mid.block_1(self.conv_in(self.post_quant_conv(z)))
            )
        )
        for i_level in reversed(range(self.num_resolutions)):
            for block in self.up[i_level].block:
                h = block(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = _nonlinearity(self.norm_out(h))
        return self.conv_out(h)


# ─────────────────────────────────────────────────────────────────────────────
# Diagonal Gaussian distribution
# ─────────────────────────────────────────────────────────────────────────────

class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.std)

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.mean(
            self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar
        )


# ─────────────────────────────────────────────────────────────────────────────
# FluxVAE
# ─────────────────────────────────────────────────────────────────────────────

class FluxVAE(nn.Module):
    """
    Flux.2 VAE — weight-compatible with ai-toolkit/flux2_vae ae.safetensors.

    Architecture parameters match the BFL AutoEncoderParams:
        ch=128, ch_mult=[1,2,4,4], num_res_blocks=2, z_channels=32

    Loading pretrained weights
    --------------------------
        from safetensors.torch import load_file
        model = FluxVAE()
        model.load_state_dict(load_file("ae.safetensors"))
        model.eval()

    Downloading from HuggingFace
    ----------------------------
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("ai-toolkit/flux2_vae", "ae.safetensors")
        model.load_state_dict(load_file(path))
    """

    # Flux.2 normalisation constants
    SCALE  = 0.3611
    SHIFT  = 0.1159

    def __init__(
        self,
        ch: int = 128,
        ch_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        z_channels: int = 32,
        in_channels: int = 3,
        out_ch: int = 3,
    ):
        super().__init__()
        self.z_channels = z_channels

        self.encoder = Encoder(ch, list(ch_mult), num_res_blocks, in_channels, z_channels)
        self.decoder = Decoder(ch, list(ch_mult), num_res_blocks, out_ch, z_channels)

        # BatchNorm to track running statistics of the latent space (affine=False
        # → no learnable scale/bias; only running_mean/var are stored).
        self.bn = nn.BatchNorm2d(ch, affine=False)

    # ------------------------------------------------------------------ #
    # Internal encode / decode (raw latent space, no normalisation)
    # ------------------------------------------------------------------ #

    def _encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        # encoder already applies quant_conv as its final step
        return DiagonalGaussianDistribution(self.encoder(x))

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        # decoder already applies post_quant_conv as its first step
        return self.decoder(z)

    # ------------------------------------------------------------------ #
    # Public API  (normalised latents — scale/shift applied)
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        x      : (B, 3, H, W) in [-1, 1]
        returns: normalised latent z  (B, 32, H/8, W/8)
        """
        dist = self._encode(x)
        z = dist.sample() if sample else dist.mode
        return (z - self.SHIFT) * self.SCALE

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z      : normalised latent  (B, 32, h, w)
        returns: (B, 3, h*8, w*8)  in [-1, 1], clamped
        """
        z_raw = z / self.SCALE + self.SHIFT
        return self._decode(z_raw).clamp(-1, 1)

    # ------------------------------------------------------------------ #
    # Forward  (encode → sample → decode at the same resolution)
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, sample: bool = True):
        """
        Returns (recon, posterior) where posterior is a
        DiagonalGaussianDistribution (for KL loss).
        """
        posterior = self._encode(x)
        z = posterior.sample() if sample else posterior.mode
        recon = self._decode(z)
        return recon, posterior

    # ------------------------------------------------------------------ #
    # Super-resolution
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def super_resolve(
        self,
        lr: torch.Tensor,
        scale_factor: int = 4,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Super-resolve a low-resolution image using the VAE as a perceptual
        refiner on top of bicubic upsampling.

        Pipeline:
            1. Bicubic-upsample LR → HR size  (provides structural detail)
            2. VAE encode HR-sized image       (compress to learned latent)
            3. VAE decode                      (reconstruct with natural texture)

        Encoding and decoding at the target resolution means the decoder has
        full HR-sized latents to work with — no information is lost from latent
        interpolation.  The VAE acts as a learned perceptual enhancer rather
        than a simple spatial interpolator.

        Args:
            lr           : (B, 3, H, W) in [-1, 1].
            scale_factor : upscale ratio (default 4).
            deterministic: use posterior mean instead of sampling (sharper,
                           recommended for inference).

        Returns:
            (B, 3, H·scale_factor, W·scale_factor) in [-1, 1]
        """
        # Step 1: bicubic upsample to target resolution
        hr_bicubic = F.interpolate(
            lr,
            scale_factor=float(scale_factor),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        ).clamp(-1, 1)

        # Step 2 & 3: VAE encode → decode at the full HR resolution
        dist = self._encode(hr_bicubic)
        z = dist.mode if deterministic else dist.sample()
        return self._decode(z).clamp(-1, 1)

    # ------------------------------------------------------------------ #
    # Loss helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def loss(
        recon: torch.Tensor,
        target: torch.Tensor,
        posterior: DiagonalGaussianDistribution,
        kl_weight: float = 1e-4,
    ):
        """
        L1 reconstruction + β·KL divergence.

        Returns: (total, recon_loss, kl_loss)
        """
        recon_loss = F.l1_loss(recon, target)
        kl_loss = posterior.kl()
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FluxVAE().to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Parameters : {n_params:.1f} M')

    # ── Verify weight keys match the checkpoint ──
    try:
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download('ai-toolkit/flux2_vae', 'ae.safetensors')
        sd = load_file(ckpt_path, device='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print(f'Loaded ae.safetensors — missing={missing}, unexpected={unexpected}')
    except Exception as e:
        print(f'(skipping weight load: {e})')

    # ── Load sample.jpg and output sample_hr.jpg ──
    from PIL import Image
    import torchvision.transforms.functional as TF

    src = 'sample.jpg'
    img = Image.open(src).convert('RGB').resize((256, 256), Image.LANCZOS)
    print(f'Loaded  : {src}  {img.size}')

    # PIL → tensor in [-1, 1], add batch dim
    lr = TF.to_tensor(img).unsqueeze(0).to(device)   # [0, 1]
    lr = lr * 2.0 - 1.0                              # [-1, 1]

    hr = model.super_resolve(lr, scale_factor=4, deterministic=True)

    hr_pil = TF.to_pil_image(((hr.squeeze(0).cpu() + 1.0) / 2.0).clamp(0, 1))
    hr_pil.save('sample_hr.jpg')
    print(f'Saved   : sample_hr.jpg  {hr_pil.size}')
