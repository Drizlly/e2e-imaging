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
    sensor_array_enabled: bool = eqx.field(static=True)
    gaussian_sigma: float = eqx.field(static=True)
    sensor_array_params: dict = eqx.field(static=True) # QUESTION: IS THIS GONNA CAUSE A PROBLEM?

    def __init__(self, 
                 photon_count = 160,
                 noise_enabled = False, 
                 sensor_array_enabled = False,
                 gaussian_sigma = None,
                 sensor_array_params = {}):
        super().__init__()
        self.photon_count = photon_count
        self.noise_enabled = noise_enabled
        self.sensor_array_enabled = sensor_array_enabled
        self.sensor_array_params = sensor_array_params
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
    
    def create_sensor_array(self, params = {}):
        if not params:
            params = self.sensor_array_params

        H, W = params.get('H', 96), params.get('W', 96)
        rows, cols = params.get('rows', 5), params.get('cols', 4)  # number of sensors vertically and horizontally
        sensor_h, sensor_w = params.get('sensor_h', 10), params.get('sensor_w', 15) # each sensor block size
        spacing_y, spacing_x = params.get('spacing_y', 8), params.get('spacing_x', 10)  # more vertical spacing between rows

        # Compute total pattern size
        pattern_h = rows * sensor_h + (rows - 1) * spacing_y
        pattern_w = cols * sensor_w + (cols - 1) * spacing_x

        # Center the pattern in the full image
        y_start = (H - pattern_h) // 2
        x_start = (W - pattern_w) // 2

        # Initialize mask
        sensor_mask = np.zeros((H, W), dtype=np.float32) 

        # Fill rectangular sensor regions
        for r in range(rows):
            for c in range(cols):
                y0 = y_start + r * (sensor_h + spacing_y)
                y1 = y0 + sensor_h
                x0 = x_start + c * (sensor_w + spacing_x)
                x1 = x0 + sensor_w
                sensor_mask[y0:y1, x0:x1] = 1.0  # active region
        
        return jnp.asarray(sensor_mask)
    
    def apply_sensor_array(self, images):
        sensor_array = self.create_sensor_array()
        assert images.shape[-2:] == sensor_array.shape, \
            f"Input images {images.shape} don't match sensor array {sensor_array.shape}"
        return images * sensor_array

    def __call__(self, images, ensure_positive=True, key=None):
        # include batch sizes
        key = jax.random.PRNGKey(0) if key is None else key
        noisy_images = images
        if self.noise_enabled:
            noisy_images = self.add_noise(images, key=key, ensure_positive=ensure_positive)
        
        masked_images = noisy_images
        if self.sensor_array_enabled:
            masked_images = self.apply_sensor_array(images)

        return masked_images