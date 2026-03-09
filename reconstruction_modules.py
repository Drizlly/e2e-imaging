import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


class ConvBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, in_channels: int, out_channels: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, key=k2)
        self.norm1 = eqx.nn.LayerNorm((out_channels,))
        self.norm2 = eqx.nn.LayerNorm((out_channels,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (C, H, W)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        return x


class UNetDeconv(eqx.Module):
    # encoder
    enc1: ConvBlock
    enc2: ConvBlock
    enc3: ConvBlock
    # bottleneck
    bottleneck: ConvBlock
    # decoder
    dec3: ConvBlock
    dec2: ConvBlock
    dec1: ConvBlock
    # final
    final_conv: eqx.nn.Conv2d

    def __init__(self, in_channels: int = 2, base_channels: int = 32, key: Optional[jax.random.PRNGKey] = None):
        key = jax.random.PRNGKey(0) if key is None else key
        k = jax.random.split(key, 8)

        c = base_channels
        self.enc1      = ConvBlock(in_channels, c,     k[0])
        self.enc2      = ConvBlock(c,           c * 2, k[1])
        self.enc3      = ConvBlock(c * 2,       c * 4, k[2])
        self.bottleneck = ConvBlock(c * 4,      c * 8, k[3])
        self.dec3      = ConvBlock(c * 8 + c * 4, c * 4, k[4])
        self.dec2      = ConvBlock(c * 4 + c * 2, c * 2, k[5])
        self.dec1      = ConvBlock(c * 2 + c,     c,     k[6])
        self.final_conv = eqx.nn.Conv2d(c, 1, kernel_size=1, key=k[7])

    def _downsample(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (C, H, W) -> (C, H/2, W/2)
        return jax.image.resize(x, (x.shape[0], x.shape[1] // 2, x.shape[2] // 2), method='nearest')

    def _upsample(self, x: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        # upsample x to match target spatial dims
        return jax.image.resize(x, (x.shape[0], target.shape[1], target.shape[2]), method='bilinear')

    def _forward_single(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (2, H, W) — single image, not batched
        # encoder
        e1 = self.enc1(x)                                               # (C, H, W)
        e2 = self.enc2(self._downsample(e1))                            # (2C, H/2, W/2)
        e3 = self.enc3(self._downsample(e2))                            # (4C, H/4, W/4)

        # bottleneck
        b = self.bottleneck(self._downsample(e3))                       # (8C, H/8, W/8)

        # decoder with skip connections
        d3 = self.dec3(jnp.concatenate([self._upsample(b, e3), e3], axis=0))   # (4C, H/4, W/4)
        d2 = self.dec2(jnp.concatenate([self._upsample(d3, e2), e2], axis=0))  # (2C, H/2, W/2)
        d1 = self.dec1(jnp.concatenate([self._upsample(d2, e1), e1], axis=0))  # (C, H, W)

        out = self.final_conv(d1)                                        # (1, H, W)
        # return out[0]       
        return out[0]                                             # (H, W)

    def __call__(self, y: jnp.ndarray, psf: jnp.ndarray) -> jnp.ndarray:
        # y:   (B, H, W)
        # psf: (H, W)
        B, H, W = y.shape

        psf_resized = jax.image.resize(psf, (H, W), method='bilinear')  # (H, W)

        # embed PSF to match image size and concatenate -> (B, 2, H, W)
        psf_embedded = jnp.broadcast_to(psf_resized[None, :, :], (B, H, W))
        x = jnp.stack([y, psf_embedded], axis=1)                        # (B, 2, H, W)

        # vmap over batch dimension
        return jax.vmap(self._forward_single)(x)                        # (B, H, W)



class WienerDeconv(eqx.Module):
    log_K: jnp.ndarray

    def __init__(self, log_K=jnp.array(-4.0)):
        self.log_K = log_K
    
    def __call__(self, y:jnp.ndarray, psf: jnp.ndarray):
        K = jnp.exp(self.log_K)  # always positive
        
        H, W = y.shape[1], y.shape[2]
        psf_h, psf_w = psf.shape

        if psf_h > H or psf_w > W:
            raise ValueError(f"PSF ({psf_h}x{psf_w}) is larger than image ({H}x{W})")

        pad_h = (psf_h - 1) // 2
        pad_w = (psf_w - 1) // 2

        psf_padded = jnp.zeros((H, W))
        psf_padded = psf_padded.at[:psf_h, :psf_w].set(psf)
        psf_padded = jnp.roll(psf_padded, (-pad_h, -pad_w), axis=(0, 1))

        PSF = jnp.fft.rfft2(psf_padded)
        Y   = jnp.fft.rfft2(y)

        PSF_conj  = jnp.conj(PSF)
        PSF_power = jnp.abs(PSF) ** 2
        wiener    = PSF_conj / (PSF_power + K)

        X_est = wiener[None, :, :] * Y

        return jnp.fft.irfft2(X_est, s=(H, W))