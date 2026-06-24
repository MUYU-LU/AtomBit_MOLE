r'''Utility package.'''

import copy
from .mixin import CastMixin
from ._scatter import scatter, group_argsort, group_cat, scatter_concat
from ._segment import segment
from .functions import cumsum, swapaxes, index_fill, index_select, broadcast_to
from .degree import degree
from .softmax import softmax
from .sort_edge_index import sort_edge_index
from .coalesce import coalesce
from .repeat import repeat
from .undirected import is_undirected, to_undirected
from .loop import (
    contains_self_loops,
    remove_self_loops,
    segregate_self_loops,
    add_self_loops,
    add_remaining_self_loops,
    get_self_loop_attr,
)
from .isolated import contains_isolated_nodes, remove_isolated_nodes
from .subgraph import get_num_hops, subgraph, k_hop_subgraph, bipartite_subgraph, hyper_subgraph
from .dropout import dropout_node, dropout_edge
from .homophily import homophily
from .assortativity import assortativity
from .laplacian import get_laplacian, get_mesh_laplacian
from .mask import mask_select, index_to_mask, mask_to_index
from .select import select, narrow
from .to_dense_batch import to_dense_batch
from .to_dense_adj import to_dense_adj
from .sparse import (
    is_sparse_tensor,
    ptr2index,
    index2ptr,
)
from .num_nodes import maybe_num_nodes
from .unbatch import unbatch, unbatch_edge_index
from .normalize import normalized_cut
from .grid import grid
from .convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from .convert import to_networkx, from_networkx
from .convert import to_trimesh, from_trimesh
from .convert import to_tensor, to_array
from .random import (
    erdos_renyi_graph,
    barabasi_albert_graph,
)
from .negative_sampling import (
    negative_sampling,
    batched_negative_sampling,
    structured_negative_sampling,
    structured_negative_sampling_feasible,
)
from .augmentation import shuffle_node, mask_feature, add_random_edge
from .tree_decomposition import tree_decomposition
from .embedding import get_embeddings
from .trim_to_layer import trim_to_layer, TrimToLayer
from .cluster import radius_graph
from .ncon import Ncon


__all__ = [
    'segment',
    'scatter',
    'group_argsort',
    'group_cat',
    'scatter_concat',
    'cumsum',
    'swapaxes',
    'index_fill',
    'index_select',
    'broadcast_to',
    'degree',
    'softmax',
    'sort_edge_index',
    'coalesce',
    'is_undirected',
    'to_undirected',
    'contains_self_loops',
    'remove_self_loops',
    'segregate_self_loops',
    'add_self_loops',
    'add_remaining_self_loops',
    'get_self_loop_attr',
    'contains_isolated_nodes',
    'remove_isolated_nodes',
    'get_num_hops',
    'subgraph',
    'bipartite_subgraph',
    'k_hop_subgraph',
    'hyper_subgraph',
    'dropout_node',
    'dropout_edge',
    'CastMixin',
    'homophily',
    'assortativity',
    'get_laplacian',
    'get_mesh_laplacian',
    'mask_select',
    'index_to_mask',
    'mask_to_index',
    'select',
    'narrow',
    'to_dense_batch',
    'to_dense_adj',
    'to_tensor',
    'to_array',
    'is_sparse_tensor',
    'index2ptr',
    'ptr2index',
    'maybe_num_nodes',
    'unbatch',
    'unbatch_edge_index',
    'normalized_cut',
    'grid',
    'to_scipy_sparse_matrix',
    'from_scipy_sparse_matrix',
    'to_networkx',
    'from_networkx',
    'to_trimesh',
    'from_trimesh',
    'erdos_renyi_graph',
    'barabasi_albert_graph',
    'negative_sampling',
    'batched_negative_sampling',
    'structured_negative_sampling',
    'structured_negative_sampling_feasible',
    'shuffle_node',
    'mask_feature',
    'add_random_edge',
    'tree_decomposition',
    'get_embeddings',
    'trim_to_layer',
    'TrimToLayer',
    'repeat',
    'radius_graph',
    'Ncon',
]

# `structured_negative_sampling_feasible` is a long name and thus destroys the
# documentation rendering. We remove it for now from the documentation:
classes = copy.copy(__all__)
classes.remove('structured_negative_sampling_feasible')
