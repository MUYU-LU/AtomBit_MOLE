r"""GNN profiling package."""

from .benchmark import benchmark
from .utils import (
    count_parameters,
    get_cpu_memory_from_gc,
    get_data_size,
    get_model_size,
)

__all__ = [
    'count_parameters',
    'get_model_size',
    'get_data_size',
    'get_cpu_memory_from_gc',
    'benchmark',
]

classes = __all__
