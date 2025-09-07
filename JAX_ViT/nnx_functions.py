import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from typing import NamedTuple, TypeAlias, Union
import jaxlib
import jaxlib.xla_extension

Array: TypeAlias = jax.Array #Union[jax.Array, np.ndarray]
PRNGKey: TypeAlias = Union[jaxlib.xla_extension.ArrayImpl, jax.random.PRNGKey] # type: ignore

get_labels = lambda data_loader: np.array([l for batch in data_loader for l in  batch[1].numpy().tolist()])

def per_layer_grad_norm(grads:nnx.statelib.State) -> dict:
    return jax.tree.map(jnp.linalg.norm, grads.to_pure_dict())

def summery_norm(parameter_dict:dict) -> tuple[Array, Array]:
    arr = jnp.array(jax.tree.flatten(parameter_dict)[0])
    return arr.mean(), arr.std()

class History(NamedTuple):
    loss: list = []
    accuracy: list = []
    LR: list = []
    gradient_norm: list = []

    def __call__(self, loss: float, accuracy: float, lr: float, grad_norms: dict) -> str:
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.LR.append(lr)
        self.gradient_norm.append(grad_norms)
        grad_norm_mean, grad_norm_std = summery_norm(grad_norms)
        list_to_fit = [
            f"Loss: {loss:.4f}",
            f"Acc: {accuracy:.4f}",
            f"LR: {lr:.6f}",
            f"grad norm mean: {grad_norm_mean:.3f}",
            f"grad norm std: {grad_norm_std:.3f}"]
        return " | ".join(list_to_fit)

# --- Training Setup ---
def compute_loss(logits: Array, targets: Array) -> Array:
    loss_terms = optax.softmax_cross_entropy_with_integer_labels(
        logits, targets
    )
    return loss_terms.mean()


def accuracy(logits: Array, targets: Array, return_pred_class:bool=False) -> Union[tuple[Array, Array], float, Array]:
    logits_argmax = logits.argmax(-1)
    mean_accuracy = (logits_argmax == targets).mean()
    out = (mean_accuracy, logits_argmax) if return_pred_class else mean_accuracy
    return out
jit_accuracy = jax.jit(lambda logits, targets: accuracy(logits, targets, False))

@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    inputs: jax.Array, targets: jax.Array
    ) -> Union[Array, tuple]:
    model.train()
    def loss_fn(model):
        logits = model(inputs)
        acc = jit_accuracy(logits, targets)
        return compute_loss(logits, targets), acc
    (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    return loss, acc, grads

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss, total_acc, total_count = 0.0, 0.0, 0
    for batch in test_loader:
        inputs, targets = batch
        inputs, targets = jnp.array(inputs.numpy()), jnp.array(targets.numpy())
        # model expects (B, N, C) style in this project convention; swap if needed
        logits = model(inputs.swapaxes(1, -1))
        loss = criterion(logits, targets)
        acc = jit_accuracy(logits, targets)

        total_loss += float(loss) * len(inputs)
        total_acc += float(acc) * len(inputs)
        total_count += len(inputs)

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return {"avg_loss": avg_loss, "avg_accuracy": avg_acc}

def stack_grad_norm_history(grad_norm_history:dict):
    """
    Given a list of nested dicts (each a grad norm snapshot), 
    return a nested dict with the same structure, but with lists/arrays
    of values for each leaf, representing the history over time.
    """
    from collections.abc import Mapping
    import numpy as np

    def recursive_stack(keys, snapshots):
        # Base case: if not a dict, stack values
        if not isinstance(snapshots[0], Mapping):
            # Stack or collect as list
            try:
                return np.stack(snapshots)
            except Exception:
                return list(snapshots)
        # Otherwise, recurse for each key
        out = {}
        for k in snapshots[0]:
            out[k] = recursive_stack(keys + [k], [snap[k] for snap in snapshots])
        return out

    return recursive_stack([], grad_norm_history)
