from typing import List, Optional, Union

import mindspore as ms
from mindspore import Tensor, ops, mint
from .functions import cumsum


from .repeat import repeat

_knn = ops.MultitypeFuncGraph('_knn')
_radius = ops.MultitypeFuncGraph('_radius')
_nearest = ops.MultitypeFuncGraph('_nearest')
_fps = ops.MultitypeFuncGraph('_fps')
_grid = ops.MultitypeFuncGraph('_grid')
_rw = ops.MultitypeFuncGraph('_rw')
_graclus = ops.MultitypeFuncGraph('_graclus')


@_knn.register('Number', 'Bool', 'Tensor', 'Tensor', 'Number', 'Number')
def _knn(k: int, cosine: bool, x: Tensor, y: Tensor, ptr_x: int = 0, ptr_y: int = 0):
    if cosine:
        dist = None
        raise NotImplementedError('The parameter cosine has not been implemented!')
    else:
        dist = ops.cdist(y, x)

    _, neighbors = mint.sort(dist, dim=1)
    neighbors = neighbors[:, :k]
    src = mint.nonzero(mint.ones_like(neighbors))[:, 0].int()
    dst = neighbors.view(-1).astype(ms.int32)
    edge_list = mint.stack([src + ptr_y, dst + ptr_x], dim=0)
    return edge_list


def knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    cosine: bool = False,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        cosine (boolean, optional): If :obj:`True`, will use the Cosine
            distance instead of the Euclidean distance to find nearest
            neighbors. (default: :obj:`False`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import mindspore as ms
        from mindspore import knn

        x = Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = ms.Tensor([0, 0, 0, 0])
        y = Tensor([[-1, 0], [1, 0]])
        batch_y = ms.Tensor([0, 0])
        assign_index = knn(x, y, 2, batch_x, batch_y)
    """
    if batch_x is not None:
        count_x = batch_x.bincount().long().astype(ms.int32)
        ptr_x = [0] + count_x.cumsum().tolist()[:-1]
        x = x.split(count_x.tolist())
    elif batch_size is not None:
        ptr_x = (mint.arange(batch_size) * batch_size).tolist()
        x = x.split(batch_size)
    else:
        ptr_x = [0]
        x = [x]
    if batch_y is not None:
        count_y = batch_y.bincount().long().astype(ms.int32)
        ptr_y = [0] + count_y.cumsum().tolist()[:-1]
        y = y.split(count_y.tolist())
    elif batch_size is not None:
        ptr_y = (mint.arange(batch_size) * batch_size).tolist()
        y = y.split(batch_size)
    else:
        ptr_y = [0]
        y = [y]
    assert len(x) == len(y)
    common_map = ops.Map()
    edge_index = common_map(ops.partial(_knn, k, cosine), x, y, ptr_x, ptr_y)
    edge_index = mint.cat(edge_index, dim=1)
    return edge_index


def knn_graph(
    x: Tensor,
    k: int,
    batch: Optional[Tensor] = None,
    loop: bool = False,
    flow: str = 'src_to_trg',
    cosine: bool = False,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Computes graph edges to the nearest :obj:`k` points.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"src_to_trg"` or
            :obj:`"trg_to_src"`). (default: :obj:`"src_to_trg"`)
        cosine (boolean, optional): If :obj:`True`, will use the Cosine
            distance instead of Euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import mindspore as ms
        from mindGeometric_cluster import knn_graph

        x = Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = ms.Tensor([0, 0, 0, 0])
        edge_index = knn_graph(x, k=2, batch=batch, loop=False)
    """

    assert flow in ['src_to_trg', 'trg_to_src']
    edge_index = knn(x, x, k if loop else k + 1, batch, batch, cosine, batch_size)

    if flow == 'src_to_trg':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return mint.stack([row, col], dim=0)


@_radius.register('Number', 'Number', 'Tensor', 'Tensor')
def _radius(r: float, max_num_neighbors: int = 32, x: Tensor = None, y: Tensor = None) -> Tensor:
    dist = ops.cdist(x, y)
    if max_num_neighbors is None:
        edge_list = mint.nonzero(dist < r)
    else:
        sorted_dist, neighbors = mint.sort(dist, dim=1)
        mask = sorted_dist < r
        if max_num_neighbors < len(x):
            neighbors = neighbors[:, :max_num_neighbors]
            mask = mask[:, :max_num_neighbors]
        src = mint.nonzero(mask)[:, 0].int()
        dst = neighbors[mask]
        edge_list = mint.stack([src, dst], dim=0)
    return edge_list


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    max_num_neighbors: int = 32,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    .. code-block:: python

        import mindspore as ms
        from mindGeometric_cluster import radius

        x = Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = ms.Tensor([0, 0, 0, 0])
        y = Tensor([[-1, 0], [1, 0]])
        batch_y = ms.Tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)
    """
    if batch_x is not None:
        batch_x = batch_x.bincount().long().tolist()
        x = x.split(batch_x)
    elif batch_size is not None:
        x = x.split(batch_size)
    else:
        x = [x]
    if batch_y is not None:
        batch_y = batch_y.bincount().long().tolist()
        y = y.split(batch_y)
    elif batch_size is not None:
        y = y.split(batch_size)
    else:
        y = [y]
    assert len(x) == len(y)
    common_map = ops.Map()
    edge_index = common_map(ops.partial(_radius, r, max_num_neighbors), x, y)
    edge_index = mint.cat(edge_index, dim=1)
    return edge_index


def radius_graph(
    x: Tensor,
    r: float,
    batch: Optional[Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "src_to_dst",
    batch_size: Optional[int] = None,


) -> Tensor:
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"src_to_dst"` or
            :obj:`"dst_to_src"`). (default: :obj:`"src_to_dst"`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import mindspore as ms
        from cluster import radius_graph

        x = Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = ms.Tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """
    assert flow in ['src_to_dst', 'dst_to_src']
    max_num_neighbors = max_num_neighbors if loop else max_num_neighbors + 1
    edge_index = radius(x, x, r=r, batch_x=batch, batch_y=batch, max_num_neighbors=max_num_neighbors,
                        batch_size=batch_size)
    if flow == 'src_to_dst':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return mint.stack([row, col], dim=0)


@_nearest.register('Tensor', 'Tensor', 'Number')
def _nearest(x: Tensor, y: Tensor, ptr: int = 0):
    dist = ops.cdist(x.float(), y.float())
    return dist.argmin(axis=1) + ptr


def nearest(
    x: Tensor,
    y: Tensor,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
) -> Tensor:
    r"""Clusters points in :obj:`x` together which are nearest to a given query
    point in :obj:`y`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import mindspore as ms
        from mindGeometric_cluster import nearest

        x = Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = ms.Tensor([0, 0, 0, 0])
        y = Tensor([[-1, 0], [1, 0]])
        batch_y = ms.Tensor([0, 0])
        cluster = nearest(x, y, batch_x, batch_y)
    """
    if batch_x is not None:
        count_x = batch_x.bincount().long()
        x = x.split(count_x.tolist())
    else:
        x = [x]
    if batch_y is not None:
        count_y = batch_y.bincount().long()
        ptr = [0] + cumsum(count_y).tolist()[:-1]
        y = y.split(count_y.tolist())
    else:
        ptr = [0]
        y = [y]
    common_map = ops.Map()
    edge_index = common_map(_nearest, x, y, ptr)
    edge_index = mint.cat(edge_index)
    return edge_index


@_graclus.register('Tensor', 'Tensor', 'Tensor')
def _graclus(rowptr, col, weight):
    pass


def graclus(
    edge_index: Tensor,
    weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None
):
    r"""A greedy clustering algorithm from the `"Weighted Graph Cuts without
    Eigenvectors: A Multilevel Approach" <http://www.cs.utexas.edu/users/
    inderjit/public_papers/multilevel_pami.pdf>`_ paper of picking an unmarked
    vertex and matching it with one of its unmarked neighbors (that maximizes
    its edge weight).
    The GPU algorithm is adapted from the `"A GPU Algorithm for Greedy Graph
    Matching" <http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf>`_
    paper.

    Args:
        edge_index (Tensor): The edge indices.
        weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    row, col = edge_index[0], edge_index[1]
    if num_nodes is None:
        num_nodes = max(int(row.max()), int(col.max())) + 1

    # Remove self-loops.
    mask = row != col
    row, col = row[mask], col[mask]

    if weight is not None:
        weight = weight[mask]

    # Randomly shuffle nodes.
    if weight is None:
        perm = ops.shuffle(mint.arange(row.size(0)))
        row, col = row[perm], col[perm]

    # To CSR.
    perm = ops.argsort(row)
    row, col = row[perm], col[perm]

    if weight is not None:
        weight = weight[perm]

    deg = mint.zeros(num_nodes).long()
    ops.tensor_scatter_elements(row, 0, mint.ones_like(row), reduction='add')
    rowptr = mint.zeros(num_nodes + 1).long()
    rowptr[1:] = mint.cumsum(deg, 0)
    return _graclus(rowptr, col, weight)


@_grid.register()
def _grid(pos, size, start, end):
    pass


def grid(pos, size, start, end):
    pass


def voxel_grid(
    pos: Tensor,
    size: Union[float, List[float], Tensor],
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, List[float], Tensor]] = None,
    end: Optional[Union[float, List[float], Tensor]] = None,
) -> Tensor:
    r"""Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
    in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
    paper, which overlays a regular grid of user-defined size over a point
    cloud and clusters all points within the same voxel.

    Args:
        pos (Tensor): Node position matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        batch (Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :attr:`pos`. (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :attr:`pos`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    dim = pos.shape[1]

    if batch is None:
        batch = mint.zeros(pos.shape[0], dtype=ms.int32)

    pos = mint.cat([pos, batch.view(-1, 1).astype(pos.dtype)], dim=-1)

    if not isinstance(size, Tensor):
        size = Tensor(size, dtype=pos.dtype)
    size = repeat(size, dim)
    size = mint.cat([size, mint.ones(1, dtype=size.dtype)])  # Add additional batch dim.

    if start is not None:
        if not isinstance(start, Tensor):
            start = Tensor(start, dtype=pos.dtype)
        start = repeat(start, dim)
        start = mint.cat([start, mint.zeros(1, dtype=start.dtype)])

    if end is not None:
        if not isinstance(end, Tensor):
            end = Tensor(end, dtype=pos.dtype)
        end = repeat(end, dim)
        end = mint.cat([end, batch.max().unsqueeze(0)])

    return grid(pos, size, start, end)


@_fps.register('Tensor', 'Number', 'Bool', 'Number')
def _fps(x: Tensor, ratio: float, random_start: bool, ptr: int):
    dist = ops.cdist(x, x)
    pos = ops.shuffle(mint.arange(len(x)))[0] if random_start else 0
    num = x.shape[0] * ratio
    num = int(num) if num > 0 else 1
    out = mint.zeros(num)
    for i in range(num):
        pos = dist[pos].argmax()
        out[num - i] = pos
        dist[:, pos] = float('nan')
    return out + ptr


def fps(x: Tensor,
        batch: Tensor,
        ratio: float = 0.5,
        random_start: bool = True,
        batch_size: int = None):
    r"""Farthest point sampling (FPS) algorithm from the 
    `"PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    .. code-block:: python

        import mindspore as ms
        from sharker.nn import fps

        x = Tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch = Tensor([0, 0, 0, 0])
        index = fps(x, batch, ratio=0.5)

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        ratio (float, optional): Sampling ratio. (default: :obj:`0.5`)
        random_start (bool, optional): If set to :obj:`False`, use the first
            node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is not None:
        count = batch.bincount().long()
        ptr = [0] + count.cumsum().tolist()[:-1]
        x = x.split(count.tolist())
    elif batch_size is not None:
        ptr = (mint.arange(batch_size) * batch_size).tolist()
        x = x.split(batch_size)
    else:
        ptr = [0]
        x = [x]
    common_map = ops.Map()
    edge_index = common_map(_fps, x, ratio, random_start, ptr)
    edge_index = mint.cat(edge_index)
    return edge_index


@_rw.register()
def _rw():
    pass


def random_walk():
    pass
