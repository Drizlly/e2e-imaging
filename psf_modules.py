import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import equinox as eqx
from typing import Tuple, Optional

import numpy as onp 
from typing import Tuple, Optional
from jax.scipy import special 
import matplotlib.pyplot as plt 


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
    grid: jnp.ndarray # cached coordinate grid  #TODO: QUESTION... should this be static??
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
        normalized_weights = jnp.clip(self.weights, a_min=1e-2)
        normalized_weights = normalized_weights / normalized_weights.sum()
        sym_covs = (self.covs + self.covs.transpose(0, 2, 1)) / 2 
        min_eigenvalue = 1e-6 # small positive constant 
        eigvals, eigvecs = jnp.linalg.eigh(sym_covs)
        eigvals = jnp.clip(eigvals, min_eigenvalue, None) # Clip eigenvalues
        min_std = 1
        eigvals = jnp.clip(eigvals, min_std**2, None) # This clip prevents super long-tailed streaks in the learned PSF. 1 works well and doesn't streak.
        new_covs = jnp.einsum("nij,nj,nkj->nik", eigvecs, eigvals, eigvecs) # TODO if any issues switch to the newer version of this? 

        # # normalize the weights 
        # normalized_weights = self.weights.clip(0) / jnp.sum(self.weights.clip(0) + 1e-10) 

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
    
# ---------------------------------------------------------------
# This code below is written by Leyla

####################
#################### Helper Functions ##########################
####################
def generate_aperture_mask(shape, use_aperture_mask):
    """ Produces either a circular aperture or a flat mask. 
    :param shape: tuple 
    :param use_aperture_mask: bool, if use_aperture_mask then makes a circular aperture. 
    """
    if not use_aperture_mask:
        return jnp.ones(shape) # return a flat unit field - multiply all by ones 
    else:
        M, N = shape 
        x, y = jnp.meshgrid(
            jnp.arange(-M // 2, M // 2), 
            jnp.arange(-N // 2, N // 2), 
            indexing='ij' 
        )
        r = jnp.sqrt(x**2 + y**2) 
        return r < ((min(M, N) // 2)) 
    # could include more than this halfway point TODO if want to shrink the radius a bit 

@jax.jit 
def apply_aperture(input_field, mask):
    return input_field * mask 

@jax.jit
def get_field_intensity(input_field):
    return jnp.abs(input_field) ** 2


def area_downsampling(input_image, downsample_factor):
    """
    Downsample an image by a specific downsample_factor

    :param input_image: Input image array, shape (height, width).
    :param target_side_length: Desired downsampling factor for the output image.
    :return: Downsampled image, shape (target_side_length, target_side_length).
    """
    input_shape = input_image.shape
    height, width = input_shape
    target_side_length = height // downsample_factor
    downsample_factor = downsample_factor

    def downsample_with_pooling(_):
        
        # take the input image 
        kernel = jnp.ones((downsample_factor, downsample_factor, 1, 1), dtype=jnp.float32) / (downsample_factor * downsample_factor) 
        pooled_image = jax.lax.conv_general_dilated( 
            input_image[None, ..., None], # add batch dimension and filler channel dimension
            kernel, 
            window_strides=(downsample_factor, downsample_factor), 
            padding="VALID", 
            dimension_numbers=("NHWC", "HWIO", "NHWC"),  # TODO check if have this wrong 
        )[0, ..., 0] # remove batch dimension and channel dimension
        
        return pooled_image 

    def downsample_with_resize(_):
        image_expanded = input_image[..., None]
        resized_image = jax.image.resize( 
            image_expanded,
            (target_side_length, target_side_length, 1), 
            method='linear'
        )
        return resized_image[..., 0] 

    # Use jax.lax.cond to handle the conditional logic
    return jax.lax.cond(
        (height % target_side_length == 0) & (width % target_side_length == 0),
        downsample_with_pooling,
        downsample_with_resize,
        operand=None
    )


####################
#################### HeightMap ##########################
####################

class HeightMap(eqx.Module):
    height_map_shape: Tuple[int, int] 
    block_size: int = 1 # enforces contiguous blocks of a particular size 
    height_map: Optional[jnp.ndarray] = None 
    blocked_height_map_shape: Tuple[int, int] 

    def __init__(self, 
                 height_map_shape: Tuple[int, int], 
                 block_size: int = 1, 
                 height_map: Optional[jnp.ndarray] = None): 
        # check that the height map is 2D 
        assert len(height_map_shape) == 2
        # check that block size divides the dimensions evenly 
        assert height_map_shape[0] % block_size == 0 and height_map_shape[1] % block_size == 0, "block size must evenly divide the height_map_shape dimensions."
        
        self.height_map_shape = height_map_shape # single channel height map 
        self.blocked_height_map_shape = [height_map_shape[0] // block_size, height_map_shape[1] // block_size] 

        if height_map is None: 
            self.height_map = jnp.ones(self.blocked_height_map_shape) * 1e-9 # initialize with flat 
        else: 
            assert height_map.shape == self.height_map_shape, "the given height_map is the wrong shape"
            self.height_map = height_map 
    def __call__(self):
        height_map = self.height_map 
        if self.block_size > 1: 
            height_map = jax.image.resize(height_map, self.height_map_shape, method='nearest') 
        # height_map = jnp.square(height_map) # instead of clipping, use squaring as a differentiable method for non-negative constraints
        height_map = jnp.where(height_map < 0.0, 0.0, height_map) # hard clip to be non-negative
        height_map = jnp.where(height_map > 4e-6, 4e-6, height_map) # hard clip to be less than 4 microns for fabrication constraints
        return height_map 
    

####################
#################### PhaseMask ##########################
####################

class PhaseMask(eqx.Module): 
    height_tolerance: float 
    lateral_tolerance: float 

    wave_length: float
    refractive_index: float 
    wave_no: float 
    delta_N: float 

    # wave_lengths = jnp.ndarray # previously done for multi-wavelength design
    # refractive_indices: jnp.ndarray 
    # wave_nos: jnp.ndarray 
    # delta_N: jnp.ndarray 

    def __init__(self, wave_length: float, refractive_index: float, 
                 height_tolerance: float=None, lateral_tolerance: float = None):
        self.wave_length = wave_length
        self.refractive_index = refractive_index
        self.height_tolerance = float(height_tolerance) if height_tolerance is not None else 0.0 
        self.lateral_tolerance = float(lateral_tolerance) if lateral_tolerance is not None else 0.0 

        self.wave_no = 2.0 * jnp.pi / wave_length 
        self.delta_N = self.refractive_index - 1.0 

    def __call__(self, input_field: jnp.ndarray, height_map: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray: 
        key = jax.random.PRNGKey(0) if key is None else key

        def add_noise(_): 
            height_error = jax.random.uniform(key, height_map.shape, minval = -self.height_tolerance, maxval = self.height_tolerance) 
            return height_map + height_error 
        height_map = jax.lax.cond( 
            self.height_tolerance > 0.0, 
            add_noise, 
            lambda _: height_map, 
            operand=None
        )

        # phase delay from height_map is 2pi/lambda * delta_N * height_map
        phi = ( 
            self.wave_no * self.delta_N * height_map
        )
        phase_delay = jnp.exp(1j * phi) 
        # apply to the input field, could have an off-axis input field for example with this model, instead of returning the delay directly
        return input_field * phase_delay 


####################
#################### Propagation Models ##########################
####################

class ASMPropagator(eqx.Module): 
    """ 
    Propagation model for shorter-length propagation. 
    """
    # shapes and scalar fields are static 
    input_shape: Tuple[int, int] # 2D, high-resolution wave array 
    distance: float # in meters 
    pixel_size: float # in meters 
    pad: int 
    pad_width: Tuple[Tuple[int, int], Tuple[int, int]] 
    padded_shape: Tuple[int, int] 

    wave_length: float 
    wave_no: float 
    H: jnp.ndarray

    def __init__(self, input_shape, distance, pixel_size, wave_length):
        self.input_shape = tuple(input_shape) # this is the original input shape to crop to 
        self.distance = distance 
        self.pixel_size = float(pixel_size) 
        self.wave_length = wave_length 
        self.wave_no = 2.0 * jnp.pi / wave_length

        assert input_shape[0] == input_shape[1]
        M_orig = input_shape[0]
        N_orig = input_shape[1] # TODO could include an assertion that it's square here, or choose the smallest? 
        pad = M_orig // 4 
        self.pad = pad 
        self.pad_width = ((pad, pad), (pad, pad)) 
        M = M_orig + 2 * pad 
        N = M_orig + 2 * pad # assuming square
        self.padded_shape = tuple((M, N))

        # frequency domain setup
        fs = 1 / self.pixel_size # frequency space bin size 
        Fx = jnp.linspace(-fs/2, fs/2, self.padded_shape[0]) 
        Fy = jnp.linspace(-fs/2, fs/2, self.padded_shape[1]) 
        Fxn, Fyn = jnp.meshgrid(Fx, Fy, indexing='ij') 
        
        # the ASM propagation kernel  # j 2pi z / lam * sqrt(1 - (lam Fxn)^2 - (lam Fyn)^2) 
        self.H = jnp.exp(2.*1j*jnp.pi*self.distance/self.wave_length * jnp.sqrt(1 - (self.wave_length * Fxn)**2 - (self.wave_length * Fyn)**2))



    def propagate(self, input_field): 
        # pad the input field 
        padded = jnp.pad(input_field, self.pad_width, mode='constant') 
        assert padded.shape == self.padded_shape, "Input field is incorrect size, please review"

        # fourier transform the padded input field 
        ft = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(padded))) # take FFT 

        # apply the propagation kernel and inverse fourier transform 
        propagated_field = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(ft * self.H)))

        # crop back to original 
        start_indices = (self.pad, self.pad) 
        cropped_propagated_field = lax.dynamic_slice(propagated_field, start_indices, self.input_shape) 
        
        return cropped_propagated_field 

    def __call__(self, input_field):
        return self.propagate(input_field) 
    
####################
#################### ArbitraryPSFLayer ##########################
####################
    

class ArbitraryPSFLayer(eqx.Module):
    """ A layer that models an arbitrary phase mask that produces a Point Spread Function using a pixel array."""

    # model input phase mask as a bunch of array values using a HeightMap wrapped inside a PhaseMask 
    # HeightMap 
    height_map_element: HeightMap # this will be an array of height values wrapped inside this class
    block_size: int # contiguous blocks inside the HeightMap
    # PhaseMask
    optical_element: PhaseMask
    refractive_index: float 
    wave_length: float 
    height_tolerance: float 
    array_upsample_resolution: int # how much bigger the array is going to be compared to the downsampled PSF shape, at least 2x upsample for avoiding aliasing 
    # Wave propagation model
    propagation_model: ASMPropagator # should be ASMPropagator in general, but offering flexibility
    sensor_to_mask_distance: float # static fixed sensor to mask distance for optimization 
    pixel_size: float 
    aperture_mask: jnp.ndarray # mask circle to concentrate to center of FOV and avoid edge effects
    object_size : int # resolution/dimensions of the object, generally 96x96 vs 32x32 
    obj_padding: tuple 
    psf_padding: tuple 
    psf_shape: tuple # static PSF shape # this is the downsampled PSF size, usually 32x32
    upsampled_psf_shape: tuple # psf_shape * array_upsample_resolution 

    measurement_bias: float # static measurement bias 
    key_seed: int
    

    def __init__(self, 
                 sensor_to_mask_distance: float,
                 refractive_index: float, 
                 wave_length: float,
                 pixel_size: float,  
                 object_size: int,
                 psf_shape: Tuple[int, int] = (32, 32), 
                 array_upsample_resolution: int = 2,
                 block_size: Optional[int] = 1, 
                 height_tolerance: Optional[float] = 0.0,
                 measurement_bias: Optional[float] = 0.0, 
                 use_aperture_mask: Optional[bool] = True, # default to including a circular mask
                 key: Optional[jax.random.PRNGKey] = None):
        super().__init__() 
        self.key_seed = 0
        self.refractive_index = refractive_index
        self.wave_length = wave_length 
        self.block_size = block_size
        self.pixel_size = pixel_size 
        self.height_tolerance = height_tolerance
        self.object_size = object_size 
        self.psf_shape = psf_shape 
        self.measurement_bias = measurement_bias 
        self.array_upsample_resolution = array_upsample_resolution 
        self.sensor_to_mask_distance = sensor_to_mask_distance 
        self.obj_padding = (0, 0)
        self.psf_padding = (0, 0) # TODO change this later, this is just for PSF and Object convolution padding

        # make sure that the input psf is square 
        assert self.psf_shape[0] == self.psf_shape[1], "Only square PSFs are supported right now."

        # first instantiate the HeightMap
        # need the high resolution dimensions 
        self.upsampled_psf_shape = tuple((psf_shape[0] * self.array_upsample_resolution, psf_shape[1] * self.array_upsample_resolution))
        self.height_map_element = HeightMap(self.upsampled_psf_shape, 
                                            block_size=self.block_size, 
                                            height_map=None)

        # then make the optical element 
        self.optical_element = PhaseMask(wave_length = self.wave_length, 
                                          refractive_index = self.refractive_index, 
                                          height_tolerance=self.height_tolerance)
        
        self.aperture_mask = generate_aperture_mask(self.upsampled_psf_shape, use_aperture_mask) 

        self.propagation_model = ASMPropagator(input_shape = self.upsampled_psf_shape,
                                               distance = self.sensor_to_mask_distance, 
                                               pixel_size = self.pixel_size, 
                                               wave_length = self.wave_length)  




    @eqx.filter_jit
    def compute_psf(self, key=None, downsample=True):
        """ 
        Compute the PSF from the current parameters. 
        Ensure energy conservation by normalizing the total sum to 1.
        """

        key = jax.random.PRNGKey(self.key_seed) if key is None else key

        # start with an input plane wave (in case you want to change it to a different input later) 
        input_field = jnp.ones(self.upsampled_psf_shape) 
        height_map = self.height_map_element() 
        psf_field = self.optical_element(input_field, height_map, key) 
        psf_field = apply_aperture(psf_field, self.aperture_mask)
        psf_field = self.propagation_model(psf_field) 
        # get intensity of field 
        psf = get_field_intensity(psf_field) 
        if not downsample: # return high resolution normalized PSF
            return psf / (jnp.sum(psf) + 1e-10) 
        # downsample to true resolution for final mask 
        psf = area_downsampling(psf, self.array_upsample_resolution)
        psf = psf / (jnp.sum(psf) + 1e-10) # normalize energy 
        return psf 
    
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
    
    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray: 
        # TODO note that this is redundant more or less with compute_psf above - it should be merged soon
        # get the PSF intensity pattern, start with an input plane wave
        key = jax.random.PRNGKey(self.key_seed) if key is None else key
        input_illumination = jnp.ones((self.upsampled_psf_shape))
        height_map = self.height_map_element() 
        psf_field = self.optical_element(input_illumination, height_map, key) 
        psf_field = apply_aperture(psf_field, self.aperture_mask)
        psf_field = self.propagation_model(psf_field) 
        psf = get_field_intensity(psf_field) 
        psf = area_downsampling(psf, self.array_upsample_resolution)
        psf = psf / (jnp.sum(psf) + 1e-10) 

        assert psf.shape == self.psf_shape 
        convolved = self.convolve2D(psf, x)
        return psf, convolved         
    
