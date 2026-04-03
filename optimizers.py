import wandb
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from typing import Optional
from scipy import ndimage


class E2EOptimizer:
    def __init__(
        self,
        model,                    # your E2E model
        lr_psf_means: float = 1e-2,
        lr_psf_covs: float = 1e-3,
        lr_psf_weights: float = 1e-4, 
        lr_recon: float = 1e-3,
        # lr = 1e-3,
        use_wandb: bool = False,
        project_name: str = 'e2e_imaging',
        run_name: str = 'run_1',
        wandb_config: dict = {},
        val_dataset: Optional[tf.data.Dataset] = None,
        test_dataset: Optional[tf.data.Dataset] = None,
        freeze_psf_covs=False,
        freeze_psf_weights=False,
    ):
        self.model = model
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        if use_wandb:
            self.wandb_config = wandb_config

        # def make_labels(params):
        #     leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
        #     labels = []
        #     for path, _ in leaves:
        #         has_log_K = any(
        #             isinstance(k, jax.tree_util.GetAttrKey) and k.name == 'log_K'
        #             for k in path
        #         )
        #         labels.append('K' if has_log_K else 'psf')
        #     return treedef.unflatten(labels)

        # separate learning rates for PSF params vs log_K
        self.optimizer = optax.multi_transform(
            {
                'psf_means': optax.adam(lr_psf_means),
                'psf_covs':    optax.set_to_zero() if freeze_psf_covs else optax.adam(lr_psf_covs),
                'psf_weights':  optax.set_to_zero() if freeze_psf_weights else optax.adam(lr_psf_weights),
                'recon':   optax.adam(lr_recon),
            },
            # label each leaf by which optimizer it should use
            param_labels=self._make_labels
        )
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))


    def _make_labels(self, model):
        params = eqx.filter(model, eqx.is_array)
        leaves, treedef = jax.tree_util.tree_flatten_with_path(params)

        labels = []
        for path, _ in leaves:
            path_str = str(path)

            if "psf_module" in path_str:
                if "means" in path_str:
                    labels.append("psf_means")
                elif "covs" in path_str:
                    labels.append("psf_covs")
                elif "weights" in path_str:
                    labels.append("psf_weights")
                else:
                    labels.append("psf_means")
            elif "reconstruction_module" in path_str:
                labels.append("recon")
            else:
                raise ValueError(f"Unrecognized parameter path: {path_str}")
        return treedef.unflatten(labels)

    def _convert_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        if hasattr(batch, "numpy"):
            batch = batch.numpy()
        return jnp.array(batch)
    
    def _compute_loss(self, dataset, key):
        total_loss = 0.0
        num_batches = 0
        for batch in dataset:
            key, subkey = jax.random.split(key)
            x_batch = self._convert_batch(batch)
            x_hat, y, psf = self.model(x_batch, subkey)
            total_loss += float(jnp.mean((x_hat - x_batch) ** 2))
            num_batches += 1
        return total_loss / num_batches

    # @eqx.filter_jit
    def step(self, model, x_batch, opt_state, key):
        def loss_fn(model):
            x_hat, y, psf = model(x_batch, key)
            return jnp.mean((x_hat - x_batch) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss

    def optimize(
        self,
        train_dataset: tf.data.Dataset,
        num_steps: int,
        log_every: int = 50,
        visualize_every: int = 200,
        key: Optional[jax.random.PRNGKey] = None
    ):
        key = jax.random.PRNGKey(0) if key is None else key

        if self.use_wandb:
            wandb.init(project=self.project_name, name=self.run_name, config=self.wandb_config)

        data_iter = iter(train_dataset)
        sample_batch = None

        for step in tqdm(range(num_steps)):
            key, subkey = jax.random.split(key)
            # get next batch, restart iterator if exhausted

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataset)
                batch = next(data_iter)


            x_batch = self._convert_batch(batch)
            if step == 0:
                sample_batch = x_batch
            self.model, self.opt_state, loss = self.step(self.model, x_batch, self.opt_state, subkey)

            # normalize PSF after each step
            self.model = eqx.tree_at(
                lambda m: m.psf_module,
                self.model,
                self.model.psf_module.normalize_psf()
            )

            if step % log_every == 0 or step == num_steps - 1:
                # K_val = float(jnp.exp(self.model.reconstruction_module.log_)) # TODO: print K when using Wiener deconv
                loss_val = float(loss)
                log_dict = {'train/loss': loss_val, 'step':step}
                # print(f"step {step}/{num_steps}  loss={loss_val:.6f}  K={K_val:.6f}")
                msg = f"step {step}/{num_steps}  loss={loss_val:.6f}"

                if self.val_dataset is not None:
                    key, subkey = jax.random.split(key)
                    val_loss = self._compute_loss(self.val_dataset, subkey)
                    log_dict['val/loss'] = val_loss
                    msg += f"  val_loss={val_loss:.6f}"

                print(msg)
                if self.use_wandb:
                    wandb.log(log_dict)

            if sample_batch is not None and (step % visualize_every == 0 or step == num_steps - 1):
                self._visualize(sample_batch, step)
        
        if self.test_dataset is not None:
            key, subkey = jax.random.split(key)
            test_loss = self._compute_loss(self.test_dataset, subkey)
            print(f"final test loss: {test_loss:.6f}")
            if self.use_wandb:
                wandb.log({'test/loss': test_loss, 'step': num_steps})
                wandb.summary['test_loss'] = test_loss

        return self.model

    def _visualize(self, x_batch, step):
        x_hat, y, psf = self.model(x_batch)

        x_np     = np.array(x_batch[0])
        y_np     = np.array(y[0])
        x_hat_np = np.array(x_hat[0])
        psf_np   = np.array(psf)
        mask = psf_np > 0.001
        labeled, num_lenslets = ndimage.label(mask)

        fig, axarr = plt.subplots(1, 5, figsize=(16, 5), constrained_layout=True)
        im0 = axarr[0].imshow(x_np,     cmap='gray'); axarr[0].set_title("x (GT)")
        im1 = axarr[1].imshow(y_np,     cmap='gray'); axarr[1].set_title("y (meas)")
        im2 = axarr[2].imshow(x_hat_np, cmap='gray'); axarr[2].set_title("x_hat")
        im3 = axarr[3].imshow(psf_np,   cmap='gray'); axarr[3].set_title("PSF")
        im4 = axarr[4].imshow(mask,     cmap='gray'); axarr[4].set_title("Mask (> 0.001)")

        for ax, im in zip(axarr, [im0, im1, im2, im3, im4]):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Step {step}")
        # plt.subplots_adjust(top=0.88, wspace=0.4) 
        
        if self.use_wandb:
            wandb.log({'viz': wandb.Image(fig), 'step': step})

        plt.show()
        plt.close(fig)

