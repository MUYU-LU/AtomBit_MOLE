from typing import Any, Dict

import numpy as np
import scipy.sparse as sp
from mindspore import Tensor, ops, nn

from ..data import Graph
from ..utils import remove_self_loops
from ..utils import to_undirected as to_undirected_fn


def read_npz(path: str, to_undirected: bool = True) -> Graph:
    with np.load(path) as f:
        return parse_npz(f, to_undirected=to_undirected)


def parse_npz(f: Dict[str, Any], to_undirected: bool = True) -> Graph:
    x = sp.csr_matrix(
        (f["attr_data"], f["attr_indices"], f["attr_indptr"]), f["attr_shape"]
    ).todense()
    x = Tensor.from_numpy(x).float()
    x[x > 0] = 1

    adj = sp.csr_matrix(
        (f["adj_data"], f["adj_indices"], f["adj_indptr"]), f["adj_shape"]
    ).tocoo()
    row = Tensor.from_numpy(adj.row).long()
    col = Tensor.from_numpy(adj.col).long()
    edge_index = ops.stack([row, col], axis=0)
    edge_index, _ = remove_self_loops(edge_index)
    if to_undirected:
        edge_index = to_undirected_fn(edge_index, num_nodes=x.shape[0])

    y = Tensor.from_numpy(f["labels"]).long()

    return Graph(x=x, edge_index=edge_index, y=y)
