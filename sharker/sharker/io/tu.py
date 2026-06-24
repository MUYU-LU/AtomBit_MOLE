import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn

from ..data import Graph
from . import fs
from .txt_array import read_txt_array
from ..utils import coalesce, cumsum, remove_self_loops

names = [
    "A",
    "graph_indicator",
    "node_labels",
    "node_attributes" "edge_labels",
    "edge_attributes",
    "graph_labels",
    "graph_attributes",
]


def read_tu_data(
    folder: str,
    prefix: str,
) -> Tuple[Graph, Dict[str, Tensor], Dict[str, int]]:
    files = fs.glob(osp.join(folder, f"{prefix}_*.txt"))
    names = [osp.basename(f)[len(prefix) + 1: -4] for f in files]

    edge_index = read_file(folder, prefix, "A", ms.int64).t() - 1
    batch = read_file(folder, prefix, "graph_indicator", ms.int64) - 1

    node_attribute = ops.zeros((batch.shape[0], 0))
    if "node_attributes" in names:
        node_attribute = read_file(folder, prefix, "node_attributes")
        if node_attribute.dim() == 1:
            node_attribute = node_attribute.unsqueeze(-1)

    node_label = ops.zeros((batch.shape[0], 0))
    if "node_labels" in names:
        node_label = read_file(folder, prefix, "node_labels", ms.int64)
        if node_label.dim() == 1:
            node_label = node_label.unsqueeze(-1)
        node_label = node_label - node_label.min(axis=0)[0]
        node_labels = node_label.unbind(axis=-1)
        node_labels = [ops.one_hot(x, x.max() + 1) for x in node_labels]
        if len(node_labels) == 1:
            node_label = node_labels[0]
        else:
            node_label = ops.cat(node_labels, axis=-1)

    edge_attribute = ops.zeros((edge_index.shape[1], 0))
    if "edge_attributes" in names:
        edge_attribute = read_file(folder, prefix, "edge_attributes")
        if edge_attribute.dim() == 1:
            edge_attribute = edge_attribute.unsqueeze(-1)

    edge_label = ops.zeros((edge_index.shape[1], 0))
    if "edge_labels" in names:
        edge_label = read_file(folder, prefix, "edge_labels", ms.int64)
        if edge_label.dim() == 1:
            edge_label = edge_label.unsqueeze(-1)
        edge_label = edge_label - edge_label.min(axis=0)[0]
        edge_labels = edge_label.unbind(axis=-1)
        edge_labels = [ops.one_hot(e, e.max() + 1) for e in edge_labels]
        if len(edge_labels) == 1:
            edge_label = edge_labels[0]
        else:
            edge_label = ops.cat(edge_labels, axis=-1)

    x = cat([node_attribute, node_label])
    edge_attr = cat([edge_attribute, edge_label])

    y = None
    if "graph_attributes" in names:  # Regression problem.
        y = read_file(folder, prefix, "graph_attributes")
    elif "graph_labels" in names:  # Classification problem.
        y = read_file(folder, prefix, "graph_labels", ms.int64)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = int(edge_index.max()) + 1 if x is None else x.shape[0]
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

    data = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    sizes = {
        "num_node_attributes": node_attribute.shape[-1],
        "num_node_labels": node_label.shape[-1],
        "num_edge_attributes": edge_attribute.shape[-1],
        "num_edge_labels": edge_label.shape[-1],
    }

    return data, slices, sizes


def read_file(
    folder: str,
    prefix: str,
    name: str,
    dtype: Optional[ms.Type] = None,
) -> Tensor:
    path = osp.join(folder, f"{prefix}_{name}.txt")
    return read_txt_array(path, sep=",", dtype=dtype)


def cat(seq: List[Optional[Tensor]]) -> Optional[Tensor]:
    values = [v for v in seq if v is not None]
    values = [v for v in values if v.numel() > 0]
    values = [v.unsqueeze(-1) if v.dim() == 1 else v for v in values]
    return ops.cat(values, axis=-1) if len(values) > 0 else None


def split(data: Graph, batch: Tensor) -> Tuple[Graph, Dict[str, Tensor]]:
    node_slice = cumsum(Tensor.from_numpy(np.bincount(batch)))

    assert data.edge_index is not None
    row, _ = data.edge_index
    edge_slice = cumsum(Tensor.from_numpy(np.bincount(batch[row])))

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {"edge_index": edge_slice}
    if data.x is not None:
        slices["x"] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = ops.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices["edge_attr"] = edge_slice
    if data.y is not None:
        assert isinstance(data.y, Tensor)
        if data.y.shape[0] == batch.shape[0]:
            slices["y"] = node_slice
        else:
            slices["y"] = ops.arange(0, int(batch[-1]) + 2, dtype=ms.int64)

    return data, slices
