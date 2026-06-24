import gc
import os
import os.path as osp
import random
import subprocess as sp
import sys
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Tuple

import mindspore as ms
from mindspore import Tensor, nn

from ..data import Data
# from ..typing import SparseTensor


def count_parameters(model: nn.Cell) -> int:
    r"""Given a :class:`nn.Cell`, count its trainable parameters.

    Args:
        model (mindspore.nn.Model): The model.
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def get_model_size(model: nn.Cell) -> int:
    r"""Given a :class:`nn.Cell`, get its actual disk size in bytes.

    Args:
        model (mindspore model): The model.
    """
    path = f'{random.randrange(sys.maxsize)}.pt'
    ms.save_checkpoint(model, path)
    model_size = osp.getsize(path)
    os.remove(path)
    return model_size


def get_data_size(data: Data) -> int:
    r"""Given a :class:`mindGeometric.data.Data` object, get its theoretical
    memory usage in bytes.

    Args:
        data (mindGeometric.data.Data or mindGeometric.data.HeteroGraph):
            The :class:`~mindGeometric.data.Data` or
            :class:`~mindGeometric.data.HeteroGraph` graph object.
    """
    data_ptrs = set()

    def _get_size(obj: Any) -> int:
        if isinstance(obj, Tensor):
            if obj in data_ptrs:
                return 0
            data_ptrs.add(obj)
            return obj.numel() * obj.element_size()
        # elif isinstance(obj, SparseTensor):
        #     return _get_size(obj.csr())
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            return sum([_get_size(x) for x in obj])
        elif isinstance(obj, Mapping):
            return sum([_get_size(x) for x in obj.values()])
        else:
            return 0

    return sum([_get_size(store) for store in data.stores])


def get_cpu_memory_from_gc() -> int:
    r"""Returns the used CPU memory in bytes, as reported by the
    :python:`Python` garbage collector.
    """
    warnings.filterwarnings('ignore', '.*mindspore.distributed.reduce_op.*')

    mem = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Tensor) and not obj.is_cuda:
                mem += obj.numel() * obj.element_size()
        except Exception:
            pass
    return mem


###############################################################################


def byte_to_megabyte(value: int, digits: int = 2) -> float:
    return round(value / (1024 * 1024), digits)


def medibyte_to_megabyte(value: int, digits: int = 2) -> float:
    return round(1.0485 * value, digits)
