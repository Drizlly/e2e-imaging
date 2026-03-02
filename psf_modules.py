import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import equinox as eqx
from typing import Tuple, Optional


class RMLPSFLayer(eqx.Module):
    """ A layer that models the Random Multi-Focal Lenslet Point Spread Function using Gaussians."""
    # model PSF layer as a bunch of gaussians 
    means: jnp.ndarray # (N, 2) array of Gaussian centers 
    covs: jnp.ndarray # (N, 2, 2) array of Gaussian covariances
    weights: jnp.ndarray # (N,) array of Gaussian weights
    object_size: int # don't make them eqx.field(static=True) it can break
    obj_padding: tuple
    psf_padding: tuple
    num_gaussians: int
    grid: jnp.ndarray # cached coordinate grid 
    psf_shape: tuple # static PSF shape (K, L)
    measurement_bias: float # static measurement bias

    def __init__(self, object_size: int, num_gaussians: int, psf_size: Tuple[int, int] = (32, 32), measurement_bias: Optional[float] = 0.0, key: Optional[jax.random.PRNGKey] = None):
        super().__init__()
        key = jax.random.PRNGKey(0) if key is None else key
        self.num_gaussians = num_gaussians
        self.object_size = object_size
        self.psf_shape = psf_size
        self.measurement_bias = measurement_bias

        # including psf_size shape to ensure it's square 
        assert psf_size[0] == psf_size[1], "PSF size must be square."

        # initialize means randomly within the PSF bounds 
        k1, k2, k3 = jax.random.split(key, 3) 
        # Create a grid of means uniformly spaced across the PSF
        grid_size = int(jnp.ceil(jnp.sqrt(num_gaussians)))
        x = jnp.linspace(-psf_size[0]/2*.7, psf_size[0]/2*.7, grid_size)
        y = jnp.linspace(-psf_size[1]/2*.7, psf_size[1]/2*.7, grid_size)
        X, Y = jnp.meshgrid(x, y)
        grid_means = jnp.stack([Y.flatten(), X.flatten()], axis=1)
        
        # Take only the number of means we need and add small random perturbations
        self.means = grid_means[:num_gaussians] + jax.random.normal(k1, (num_gaussians, 2)) * (psf_size[0]/grid_size/4)

        # self.means = jax.random.uniform(k1, (num_gaussians, 2), 
        #                             minval = -psf_size[0] // 4,
        #                             maxval = psf_size[0] // 4) # TODO other version uses //3 for spectral, decide which 
        # initialize covariance matrices with random rotation and scale 
        single_scale = jax.random.uniform(k2, (num_gaussians,), minval=1, maxval=5)
        scales = jnp.stack([single_scale, single_scale], axis=1)
        thetas = jax.random.uniform(k3, (num_gaussians,), minval=0, maxval=2*jnp.pi)

        # construct rotation matrices 
        # cos_t = jnp.cos(thetas)
        # sin_t = jnp.sin(thetas)
        # R = jnp.stack([jnp.stack([cos_t, -sin_t], axis=1),
        #             jnp.stack([sin_t, cos_t], axis=1)], axis=1)
        
        # create diagonal matrices with scales 
        # S = jnp.zeros((num_gaussians, 2, 2))
        # S = S.at[:, 0, 0].set(scales[:, 0])
        # S = S.at[:, 1, 1].set(scales[:, 1])

         # Compute covariance matrices: R @ S @ R.T
        # self.covs = jnp.einsum('nij,njk,nkl->nil', R, S, R.transpose(0, 2, 1)) # TODO newer version uses matmul but I like einsum
        self.covs = jnp.stack([jnp.eye(2) * s for s in single_scale])
        # initialize weights uniformly 
        self.weights = jnp.ones(num_gaussians) / num_gaussians

        # padding calculations # TODO skipping this, assume 2D images that are hxw and images are >> object. 
        self.obj_padding = (0, 0) # (H, W format)
        self.psf_padding = (0, 0)

        # create a coordinate grid 
        y = jnp.linspace(-psf_size[0] // 2, psf_size[0] // 2, psf_size[0])
        x = jnp.linspace(-psf_size[1] // 2, psf_size[1] // 2, psf_size[1])
        X, Y = jnp.meshgrid(x, y)
        self.grid = jnp.stack([Y.flatten(), X.flatten()], axis=1)

    def compute_psf(self):
        """
        Compute the PSF from the current parameters.
        Ensures energy conservation by normalizing the total sum to 1.
        Returns a square PSF array.
        """
        # Reshape grid for broadcasting
        grid_expanded = self.grid[None, :, :] # (1, H*W, 2)
        means_expanded = self.means[:, None, :] # (N, 1, 2) 

        # Center the coordinates 
        centered = grid_expanded - means_expanded # (N, H*W, 2) 
    
        # Compute quadratic form for each Gaussian 
        covs_inv = jnp.linalg.inv(self.covs) # (N, 2, 2)
        quad_form = jnp.einsum('npi,nij,npj->np', centered, covs_inv, centered) # TODO newer version uses a different operation. 
        # also including a clip from newer version
        quad_form_clipped = jnp.clip(quad_form, a_min=-100, a_max=100)

        # Compute Gaussian values 
        gaussians = jnp.exp(-0.5 * quad_form_clipped) # (N, H*W) 

        # Normalize each gaussian individually to ensure it integrates to 1 
        gaussians = gaussians / (jnp.sum(gaussians, axis=1, keepdims=True) + 1e-10) 

        # Weight and sum the Gaussians 
        weighted_sum = jnp.sum(self.weights[:, None] * gaussians, axis=0)

        # reshape back to square 2D array 
        psf = weighted_sum.reshape(self.psf_shape[0], self.psf_shape[1]) 

        # Normalize to conserve energy
        return psf / (jnp.sum(psf) + 1e-10)
    
    def normalize_psf(self):
        """ Ensure weights sum to 1 and that covariance matrices are positive definite."""
        # ensure positive definite 
        sym_covs = (self.covs + self.covs.transpose(0, 2, 1)) / 2 
        min_eigenvalue = 1e-6 # small positive constant 
        eigvals, eigvecs = jnp.linalg.eigh(sym_covs)
        eigvals = jnp.clip(eigvals, min_eigenvalue, None) # Clip eigenvalues
        min_std = 1
        eigvals = jnp.clip(eigvals, min_std**2, None) # This clip prevents super long-tailed streaks in the learned PSF. 1 works well and doesn't streak.
        new_covs = jnp.einsum("nij,nj,nkj->nik", eigvecs, eigvals, eigvecs) # TODO if any issues switch to the newer version of this? 

        # normalize the weights 
        normalized_weights = self.weights.clip(0) / jnp.sum(self.weights.clip(0) + 1e-10) 

        # constrain the means to be within the PSF bounds 
        means = jnp.clip(self.means, -self.psf_shape[0] // 2*.8, self.psf_shape[0] // 2*.8)

        return eqx.tree_at(
            lambda layer: (layer.covs, layer.weights, layer.means),
            self,
            (new_covs, normalized_weights, means),
        )
    
    def convolve2D(self, psf: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        psf = psf[::-1, ::-1]
        K, L = psf.shape

        # Pad fully so conv_valid gives the 'full' result
        x_padded = jnp.pad(x, ((0,0), (K-1, K-1), (L-1, L-1)))
        x4 = x_padded[:, :, :, None]
        k4 = psf[:, :, None, None]

        y4 = lax.conv_general_dilated(
            lhs=x4,
            rhs=k4,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        # y4 shape is now (B, H+K-1, W+L-1, 1) — the full convolution output

        # Crop to original size, matching scipy's 'same' anchor
        crop_h = (K - 1) // 2
        crop_w = (L - 1) // 2
        H, W = x.shape[1], x.shape[2]

        return y4[:, crop_h:crop_h+H, crop_w:crop_w+W, 0]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        psf = self.compute_psf()
        return psf, self.convolve2D(psf, x) 
