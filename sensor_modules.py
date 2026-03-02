import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import equinox as eqx
from typing import Tuple, Optional

class SensorModule(eqx.Module):
    photon_count: int = eqx.field(static=True)
    noise_enabled: bool = eqx.field(static=True)
    gaussian_sigma: float = eqx.field(static=True)

    def __init__(self, 
                 photon_count = 160,
                 noise_enabled = False, 
                 gaussian_sigma = None):
        super().__init__()
        self.photon_count = photon_count
        self.noise_enabled = noise_enabled
        self.gaussian_sigma = gaussian_sigma

    def add_noise(self, images, key=None, ensure_positive=True):
        # Scale to photon counts
        lam = images * self.photon_count
        
        if self.gaussian_sigma is not None:
            noise = jax.random.normal(key, shape=images.shape) * self.gaussian_sigma
            noisy_images = (lam + noise) / self.photon_count
        else:
            # jax.random.poisson generates a unique value for EVERY pixel automatically
            noisy_images = jax.random.poisson(key, lam=lam).astype(jnp.float32)
            noisy_images = noisy_images / self.photon_count

        if ensure_positive:
            noisy_images = jnp.maximum(0, noisy_images)
        
        return noisy_images
    
    def __call__(self, images, ensure_positive=True, key=None):
        # include batch sizes
        key = jax.random.PRNGKey(0) if key is None else key
        noisy_images = images
        if self.noise_enabled:
            noisy_images = self.add_noise(images, key=key, ensure_positive=ensure_positive)

        return noisy_images