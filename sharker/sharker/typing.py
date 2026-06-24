import os
import mindspore as ms
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from mindspore import Tensor

WITH_WINDOWS = os.name == 'nt'

MAX_INT64 = np.iinfo(np.int64).max


# Types for accessing data ####################################################

# Node-types are denoted by a single string, e.g.: `data['paper']`:
NodeType = str

# Edge-types are denotes by a triplet of strings, e.g.:
# `data[('author', 'writes', 'paper')]
EdgeType = Tuple[str, str, str]

NodeOrEdgeType = Union[NodeType, EdgeType]

DEFAULT_REL = 'to'
EDGE_TYPE_STR_SPLIT = '__'


WITH_SPARSE = False
WITH_SOFTMAX = False


class SparseStorage:  # type: ignore
    def __init__(
        self,
        row: Optional[Tensor] = None,
        rowptr: Optional[Tensor] = None,
        col: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        rowcount: Optional[Tensor] = None,
        colptr: Optional[Tensor] = None,
        colcount: Optional[Tensor] = None,
        csr2csc: Optional[Tensor] = None,
        csc2csr: Optional[Tensor] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        raise ImportError("'SparseStorage' requires 'torch-sparse'")

    def value(self) -> Optional[Tensor]:
        raise ImportError("'SparseStorage' requires 'torch-sparse'")

    def rowcount(self) -> Tensor:
        raise ImportError("'SparseStorage' requires 'torch-sparse'")


class SparseTensor:  # type: ignore
    def __init__(
        self,
        row: Optional[Tensor] = None,
        rowptr: Optional[Tensor] = None,
        col: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    @classmethod
    def from_edge_index(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ) -> 'SparseTensor':
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    @property
    def storage(self) -> SparseStorage:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    @classmethod
    def from_dense(self, mat: Tensor,
                   has_value: bool = True) -> 'SparseTensor':
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def size(self, dim: int) -> int:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def nnz(self) -> int:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def is_cuda(self) -> bool:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def has_value(self) -> bool:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def set_value(self, value: Optional[Tensor],
                  layout: Optional[str] = None) -> 'SparseTensor':
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def fill_value(self, fill_value: float,
                   dtype: Optional[ms.Type] = None) -> 'SparseTensor':
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def coo(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def csr(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def requires_grad(self) -> bool:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def to_torch_sparse_csr_tensor(
        self,
        dtype: Optional[ms.Type] = None,
    ) -> Tensor:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")


class EdgeTypeStr(str):
    r"""A helper class to construct serializable edge types by merging an edge
    type tuple into a single string.
    """
    def __new__(cls, *args: Any) -> 'EdgeTypeStr':
        if isinstance(args[0], (list, tuple)):
            # Unwrap `EdgeType((src, rel, dst))` and `EdgeTypeStr((src, dst))`:
            args = tuple(args[0])

        if len(args) == 1 and isinstance(args[0], str):
            arg = args[0]  # An edge type string was passed.

        elif len(args) == 2 and all(isinstance(arg, str) for arg in args):
            # A `(src, dst)` edge type was passed - add `DEFAULT_REL`:
            arg = EDGE_TYPE_STR_SPLIT.join((args[0], DEFAULT_REL, args[1]))

        elif len(args) == 3 and all(isinstance(arg, str) for arg in args):
            # A `(src, rel, dst)` edge type was passed:
            arg = EDGE_TYPE_STR_SPLIT.join(args)

        else:
            raise ValueError(f"Encountered invalid edge type '{args}'")

        return str.__new__(cls, arg)

    def to_tuple(self) -> EdgeType:
        r"""Returns the original edge type."""
        out = tuple(self.split(EDGE_TYPE_STR_SPLIT))
        if len(out) != 3:
            raise ValueError(f"Cannot convert the edge type '{self}' to a "
                             f"tuple since it holds invalid characters")
        return out


# There exist some short-cuts to query edge-types (given that the full triplet
# can be uniquely reconstructed, e.g.:
# * via str: `data['writes']`
# * via Tuple[str, str]: `data[('author', 'paper')]`
QueryType = Union[NodeType, EdgeType, str, Tuple[str, str]]

Metadata = Tuple[List[NodeType], List[EdgeType]]

# A representation of a feature tensor
FeatureTensorType = Union[Tensor, np.ndarray]

# A representation of an edge index, following the possible formats:
#   * COO: (row, col)
#   * CSC: (row, colptr)
#   * CSR: (rowptr, col)
EdgeTensorType = Tuple[Tensor, Tensor]

# Types for message passing ###################################################

Adj = Union[Tensor, ]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]

MaybeHeteroNodeTensor = Union[Tensor, Dict[NodeType, Tensor]]
MaybeHeteroAdjTensor = Union[Tensor, Dict[EdgeType, Adj]]
MaybeHeteroEdgeTensor = Union[Tensor, Dict[EdgeType, Tensor]]

# Types for sampling ##########################################################

InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]
