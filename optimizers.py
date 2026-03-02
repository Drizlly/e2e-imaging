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


class E2EOptimizer:
    def __init__(
        self,
        model,                    # your E2E model
        # lr_psf: float = 1e-3,    # learning rate for PSF parameters
        # lr_K: float = 1e-3,      # learning rate for K
        lr = 1e-3,
        use_wandb: bool = True,
        project_name: str = 'e2e_imaging',
        run_name: str = 'run_1',
        wandb_config: dict = {}

    ):
        self.model = model
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        if use_wandb:
            self.wandb_config = wandb_config

        # separate learning rates for PSF params vs log_K
        # self.optimizer = optax.multi_transform(
        #     {
        #         'psf': optax.adam(lr_psf),
        #         'K':   optax.adam(lr_K),
        #     },
        #     # label each leaf by which optimizer it should use
        #     param_labels=self._make_labels(model)
        # )
        self.optimizer = optax.adam(lr) #TODO: do i need two learning rates, one for PSF, and one for wiener?
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

    def _make_labels(self, model):
        # returns a pytree with same structure as model,
        # where each leaf is labeled 'psf' or 'K'
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: 'K' if 'log_K' in str(path) else 'psf',
            eqx.filter(model, eqx.is_array)
        )

    def _convert_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        if hasattr(batch, "numpy"):
            batch = batch.numpy()
        return jnp.array(batch)

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
                K_val = float(jnp.exp(self.model.log_K))
                loss_val = float(loss)
                print(f"step {step}/{num_steps}  loss={loss_val:.6f}  K={K_val:.6f}")

                if self.use_wandb:
                    wandb.log({'loss': loss_val, 'K': K_val, 'step': step})

            if sample_batch is not None and (step % visualize_every == 0 or step == num_steps - 1):
                self._visualize(sample_batch, step)

        if self.use_wandb:
            K_val = float(jnp.exp(self.model.log_K))
            loss_val = float(loss)

            if self.use_wandb:
                wandb.log({'loss': loss_val, 'K': K_val, 'step': step})

            self._visualize(sample_batch, step)
            wandb.finish()

        return self.model

    def _visualize(self, x_batch, step):
        x_hat, y, psf = self.model(x_batch)

        x_np     = np.array(x_batch[0])
        y_np     = np.array(y[0])
        x_hat_np = np.array(x_hat[0])
        psf_np   = np.array(psf)

        fig, axarr = plt.subplots(1, 4, figsize=(16, 4))
        axarr[0].imshow(x_np,     cmap='gray'); axarr[0].set_title("x (GT)")
        axarr[1].imshow(y_np,     cmap='gray'); axarr[1].set_title("y (meas)")
        axarr[2].imshow(x_hat_np, cmap='gray'); axarr[2].set_title("x_hat")
        axarr[3].imshow(psf_np,   cmap='gray'); axarr[3].set_title("PSF")

        plt.suptitle(f"Step {step}")
        plt.tight_layout()
        print("CORNER VAL:", y_np[31][31] )

        if self.use_wandb:
            wandb.log({'viz': wandb.Image(fig), 'step': step})

        plt.show()
        plt.close(fig)