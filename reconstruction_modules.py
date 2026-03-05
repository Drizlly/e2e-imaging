import jax
import jax.numpy as jnp
import equinox as eqx


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