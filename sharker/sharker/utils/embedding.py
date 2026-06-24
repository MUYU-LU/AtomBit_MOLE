import warnings
from typing import Any, List

from mindspore import nn
from mindspore import Tensor, ops, nn


def get_embeddings(
    model: nn.Cell,
    *args: Any,
    **kwargs: Any,
) -> List[Tensor]:
    """Returns the output embeddings of all
    :class:`~sharker.nn.conv.MessagePassing` layers in
    :obj:`model`.

    Internally, this method registers forward hooks on all
    :class:`~sharker.nn.conv.MessagePassing` layers of a :obj:`model`,
    and runs the forward pass of the :obj:`model` by calling
    :obj:`model(*args, **kwargs)`.

    Args:
        model (nn.Cell): The message passing model.
        *args: Arguments passed to the model.
        **kwargs (optional): Additional keyword arguments passed to the model.
    """
    from ..nn import MessagePassing

    embeddings: List[Tensor] = []

    def hook(model: nn.Cell, inputs: Any, outputs: Any) -> None:
        # Clone output in case it will be later modified in-place:
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        assert isinstance(outputs, Tensor)
        embeddings.append(outputs.copy())

    hook_handles = []
    for module in model.cells():  # Register forward hooks:
        if isinstance(module, MessagePassing):
            hook_handles.append(module.register_forward_hook(hook))

    if len(hook_handles) == 0:
        warnings.warn("The 'model' does not have any 'MessagePassing' layers")

    training = model.training
    model.set_train(False)
    # model.eval()
    model(*args, **kwargs)
    model.set_train(training)

    for handle in hook_handles:  # Remove hooks:
        handle.remove()

    return embeddings
