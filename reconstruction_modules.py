import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple

LN_EPS = 1e-4

# ADAPTED FROM LENSLESS LEARNING PAPER: PUT LINK

class ConvLnRelu2d(eqx.Module):
    conv: eqx.nn.Conv2d
    ln: eqx.nn.LayerNorm
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, key=None):
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, use_bias=False, key=key)
        # self.bn = eqx.nn.BatchNorm(out_channels, axis_name="batch", eps=BN_EPS)
        self.ln = eqx.nn.LayerNorm(out_channels, eps=LN_EPS) 
    
    def __call__(self, x):
        x = self.conv(x)
        # x, state = self.bn(x, state, inference=inference)
        # x = jax.vmap(self.ln)(x)  # apply layernorm across spatial dims
        x = jax.vmap(jax.vmap(self.ln, in_axes=1, out_axes=1), in_axes=1, out_axes=1)(x) # TODO: can you do this vmap in oen go
        x = jax.nn.relu(x)
        return x

class StackEncoder(eqx.Module):
    conv1: ConvLnRelu2d
    conv2: ConvLnRelu2d

    def __init__(self, in_channels, out_channels, kernel_size=3, key=None):
        padding = (kernel_size - 1) // 2
        k1, k2 = jax.random.split(key)
        self.conv1 = ConvLnRelu2d(in_channels, out_channels, kernel_size, padding, key=k1) # TODO: look into why this needs a key!
        self.conv2 = ConvLnRelu2d(out_channels, out_channels, kernel_size, padding, key=k2)

    def __call__(self, x): #TODO: what does inference do?
        x = self.conv1(x)
        x = self.conv2(x)
        x_small = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)
        return x, x_small

class StackDecoder(eqx.Module):
    conv1: ConvLnRelu2d
    conv2: ConvLnRelu2d
    conv3: ConvLnRelu2d

    def __init__(self, in_channels, out_channels, kernel_size=3, key=None):
        padding = (kernel_size - 1) // 2
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = ConvLnRelu2d(in_channels, out_channels, kernel_size, padding, key=k1)
        self.conv2 = ConvLnRelu2d(out_channels, out_channels, kernel_size, padding, key=k2)
        self.conv3 = ConvLnRelu2d(out_channels, out_channels, kernel_size, padding, key=k3)

    def __call__(self, x, down_tensor):
        _, height, width = down_tensor.shape  # (C, H, W) in eqx

        # Upsample
        x = jax.image.resize(x, shape=(x.shape[0], height, width), method='bilinear')

        # Concatenate skip connection
        x = jnp.concatenate([x, down_tensor], axis=0)  # channel dim is 0 in eqx

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class UNetDeconv(eqx.Module):
    down1: StackEncoder
    down2: StackEncoder
    down3: StackEncoder
    down4: StackEncoder
    down5: StackEncoder
    center: ConvLnRelu2d
    up5: StackDecoder
    up4: StackDecoder
    up3: StackDecoder
    up2: StackDecoder
    up1: StackDecoder
    classify: eqx.nn.Conv2d

    def __init__(self, key=None):
        keys = jax.random.split(key, 12)
        self.down1 = StackEncoder(1,   24,  kernel_size=3, key=keys[0])
        self.down2 = StackEncoder(24,  64,  kernel_size=3, key=keys[1])
        self.down3 = StackEncoder(64,  128, kernel_size=3, key=keys[2])
        self.down4 = StackEncoder(128, 256, kernel_size=3, key=keys[3])
        self.down5 = StackEncoder(256, 512, kernel_size=3, key=keys[4])
        self.center = ConvLnRelu2d(512, 512, kernel_size=3, padding=1, key=keys[5])
        self.up5 = StackDecoder(512+512, 256, kernel_size=3, key=keys[6])
        self.up4 = StackDecoder(256+256, 128, kernel_size=3, key=keys[7])
        self.up3 = StackDecoder(128+128, 64,  kernel_size=3, key=keys[8])
        self.up2 = StackDecoder(64+64,   24,  kernel_size=3, key=keys[9])
        self.up1 = StackDecoder(24+24,   24,  kernel_size=3, key=keys[10])
        self.classify = eqx.nn.Conv2d(24, 1, kernel_size=1, use_bias=True, key=keys[11])
    
    def _single_forward(self, x):
        x = x[None, ...]
        down1, out = self.down1(x)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out  = self.down4(out)
        down5, out = self.down5(out)

        out = self.center(out)

        out = self.up5(out, down5)
        out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)
        out = jnp.squeeze(out, axis=0)
        return out

    def __call__(self, x, psf):
        return jax.vmap(self._single_forward)(x)

class UNetDeconv_small(eqx.Module):
    down1: StackEncoder
    center: ConvLnRelu2d
    up1: StackDecoder
    classify: eqx.nn.Conv2d

    def __init__(self, key=None):
        keys = jax.random.split(key, 4)
        self.down1 = StackEncoder(1,   24,  kernel_size=3, key=keys[0])
        self.center = ConvLnRelu2d(24, 24, kernel_size=3, padding=1, key=keys[1])
        self.up1 = StackDecoder(24+24,   24,  kernel_size=3, key=keys[2])
        self.classify = eqx.nn.Conv2d(24, 1, kernel_size=1, use_bias=True, key=keys[3])
    
    def _single_forward(self, x):
        x = x[None, ...]
        down1, out = self.down1(x)
        out = self.center(out)
        out = self.up1(out, down1)
        out = self.classify(out)
        out = jnp.squeeze(out, axis=0)
        return out

    def __call__(self, x, psf):
        return jax.vmap(self._single_forward)(x)


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