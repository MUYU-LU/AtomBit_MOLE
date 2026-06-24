import os.path as osp
import warnings
from itertools import repeat
from typing import Dict, List, Optional

import fsspec
import mindspore as ms
from mindspore import ops
from mindspore import Tensor, ops, nn

from ..data import Graph
from .txt_array import read_txt_array
from ..utils import (
    coalesce,
    index_to_mask,
    remove_self_loops,
)

import pickle



def read_file(folder: str, prefix: str, name: str) -> Tensor:
    path = osp.join(folder, f"ind.{prefix.lower()}.{name}")

    if name == "test.index":
        return read_txt_array(path, dtype=ms.int64)

    with fsspec.open(path, "rb") as f:
        warnings.filterwarnings("ignore", ".*`scipy.sparse.csr` name.*")
        out = pickle.load(f, encoding="latin1")

    if name == "graph":
        return out

    out = out.todense() if hasattr(out, "todense") else out
    out = Tensor.from_numpy(out).float()
    return out


def edge_index_from_dict(
    graph_dict: Dict[int, List[int]],
    num_nodes: Optional[int] = None,
) -> Tensor:
    rows: List[int] = []
    cols: List[int] = []
    for key, value in graph_dict.items():
        rows += repeat(key, len(value))
        cols += value
    row = Tensor(rows)
    col = Tensor(cols)
    edge_index = ops.stack([row, col], axis=0)

    # NOTE: There are some duplicated edges and self loops in the datasets.
    #       Other implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes, sort_by_row=False)

    return edge_index
