# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: waller_env
#     language: python
#     name: python3
# ---

# %%
import os
import sys
import optax
# set gpu to be pci bus id
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# set gpu memory usage and turnoff pre-allocated memory TODO: what does the following do??
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# %%
import jax
from jax import random
from jax import lax

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


# %%
# sys.path.append('/home/jmathew_waller/workspace/e2e-imaging/')
from psf_modules import RMLPSFLayer
from lensless_data_generator import LenslessDataGenerator
from optimizers import E2EOptimizer
from sensor_modules import SensorModule
from reconstruction_modules import WienerDeconv, UNetDeconv, UNetDeconv_small


# %%
class E2E(eqx.Module):
    psf_module: eqx.Module
    sensor_module: eqx.Module
    reconstruction_module: eqx.Module

    def __init__(self, 
                 psf_module,
                 sensor_module,
                 reconstruction_module):
        
        self.psf_module = psf_module
        self.sensor_module = sensor_module
        self.reconstruction_module = reconstruction_module
        

    def __call__(self, x: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None, ensure_positive=True) -> tuple:
        # x: (B, H, W)
        key = jax.random.PRNGKey(0) if key is None else key # TODO: ask claude, is this bad seeding?
        psf, y = self.psf_module(x)  # psf:(K, L), y:(B, H, W)
        noisy_y = self.sensor_module(y, key=key, ensure_positive=ensure_positive)
        x_hat = self.reconstruction_module(noisy_y, psf) # (B, H, W)
        
        return x_hat, noisy_y, psf

# %%
# general
lr_psf = 1e-2
lr_recon = 1e-4
seed_value = 10 #TODO: FIX SEEDING
key = jax.random.PRNGKey(seed_value)

# loading images
tile_rows=3
tile_cols=3
batch_size=32  
dataset_name = 'mnist'
photon_count = 160
subset_fraction = 0.9

# psf stuff constants
psf_size = (32, 32)
object_size = 32
num_gaussians = 1
use_ideal = False

# sensor stuff
noise_enabled = True
sensor_array_enabled = True
gaussian_sigma = 0.5

sensor_array_params = {
    "H": 96,
    "W": 96, 
    "rows": 3,
    "cols": 3,
    "sensor_h": 15,
    "sensor_w": 20,
    "spacing_y": 15,
    "spacing_x": 12
}

# recon stuff
recon_name = 'unet_small'
log_K = jnp.array(-4.0) #initial starting K value for wiener deconv

# train stuff
num_steps = 10000
visualize_every = 500

#wandb logging stuff
use_wandb = False
# project_name = 'e2e_imaging_playground'
project_name = 'e2e_fine_tuning_playground'
run_name = f'{dataset_name}_recon_{recon_name}_ideal_init_{use_ideal}_lr_recon_{lr_recon}_gaussian_sigma_{gaussian_sigma}_photon_count_{photon_count}_num_gaussian_{num_gaussians}'
log_every = 50

# %%
# set up wandb config
wandb_config = {
    'general': {
        'seed_value': seed_value,
        'num_steps': num_steps
    },
    
    'dataset': {
        'subset_fraction': subset_fraction,
        'photon_count': photon_count,
        'tile_rows': tile_rows,
        'tile_cols': tile_cols,
        'batch_size': batch_size,
        'dataset_name': dataset_name,
    },
    
    'psf_layer_module': {
        'object_size': object_size,
        'num_gaussians': num_gaussians,
        'psf_size': psf_size,
        'gaussian_sigma': gaussian_sigma,
        'use_ideal': use_ideal
    },

    'sensor_module': {
        'noise_enabled': noise_enabled,
        'sensor_array_enabled': sensor_array_enabled, 
        'gaussian_sigma': gaussian_sigma,
        'sensor_array_params': sensor_array_params
    },

    'reconstruction_module': {
        'initial log_K': log_K,
        'learning_rate_psf': lr_psf,
        'learning_rate_recon': lr_recon
    },
    
    'logging': {
        'use_wandb': use_wandb,
        'project_name': project_name,
        'run_name': run_name,
        'log_every': log_every,
        'visualize_every': visualize_every
    }
}

# %%
# load images!
data_generator = LenslessDataGenerator(photon_count, subset_fraction=subset_fraction, seed=seed_value)

if dataset_name == 'cifar10':
    x_data, x_test = data_generator.load_cifar10_data()
elif dataset_name == 'mnist':
    x_data, x_test = data_generator.load_mnist_data()

x_train, x_val = train_test_split(x_data, test_size=0.1, random_state=seed_value)
train_dataset = data_generator.create_dataset(x_train, tile_rows=tile_rows, tile_cols=tile_cols, batch_size=batch_size)
val_dataset = data_generator.create_dataset(x_val, tile_rows=tile_rows, tile_cols=tile_cols, batch_size=batch_size)
test_dataset =  data_generator.create_dataset(x_test, tile_rows=tile_rows, tile_cols=tile_cols, batch_size=batch_size)


print("x_train range:", x_train.min(), x_train.max(), x_train.mean())
print("x_val range:",   x_val.min(),   x_val.max(),   x_val.mean())
print("x_test range:",  x_test.min(),  x_test.max(),  x_test.mean())
x_train[0].shape[0] * tile_rows


# %%
# the model (psf layer)
key, subkey = jax.random.split(key) # TODO: replace this with self.next_rng_key()??
psf_module = RMLPSFLayer(object_size=96, num_gaussians=num_gaussians, psf_size=psf_size, key=subkey)

if use_ideal:
    params = np.load("/home/juliama/wallerlab/e2e-imaging/ideal_psfs/optimized_psf_params.npz")

    psf_module = eqx.tree_at(
        lambda m: (m.means, m.covs, m.weights),
        psf_module,
        (
            jnp.array(params['means']),
            jnp.array(params['covs']),
            jnp.array(params['weights'])
        )
    )

# %%

# psf_module = RMLPSFLayer(object_size=32, num_gaussians=10, psf_size=psf_size, key=subkey_1)
sensor_module = SensorModule(photon_count=photon_count, 
                             noise_enabled=noise_enabled,
                             sensor_array_enabled=sensor_array_enabled, 
                             gaussian_sigma=gaussian_sigma, 
                             sensor_array_params= sensor_array_params)


# %%
@eqx.filter_jit
def train_step(model, x_batch, opt_state, optimizer, key):
    def loss_fn(model):
        x_hat, y, psf = model(x_batch, key)
        return jnp.mean((x_hat - x_batch) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, new_opt_state, loss


# %%
for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
    # reinitialize model and optimizer
    key, subkey = jax.random.split(key)
    reconstruction_module = UNetDeconv_small(key=subkey)  # fresh UNet weights
    model_test = E2E(psf_module=psf_module, sensor_module=sensor_module, reconstruction_module=reconstruction_module)
    
    optimizer_test = optax.adam(lr)
    opt_state_test = optimizer_test.init(eqx.filter(model_test, eqx.is_array))

    
    losses = []
    data_iter = iter(train_dataset)
    for step in range(650):
        key, subkey = jax.random.split(key)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataset)
            batch = next(data_iter)
        x_batch = jnp.array(batch.numpy())
        model_test, opt_state_test, loss = train_step(model_test, x_batch, opt_state_test, optimizer_test, subkey)
        losses.append(float(loss))
    
    print(f'lr_{lr}_complete')
    plt.plot(losses, label=f'lr={lr}')

plt.legend()
plt.xlabel('step')
plt.ylabel('loss')
plt.title('Loss curves for different LRs')
plt.savefig('lr_range_test.png', dpi=150, bbox_inches='tight')
plt.show()
