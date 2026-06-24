from typing import Optional, Tuple

import mindspore as ms
from mindspore import Tensor, ops, nn, mint

from .loop import add_self_loops, remove_self_loops
from . import scatter
from .num_nodes import maybe_num_nodes
from .undirected import to_undirected
from .ncon import Ncon


def get_laplacian(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    normalization: Optional[str] = None,
    dtype: Optional[ms.Type] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the graph Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        dtype (ms.Type, optional): The desired data type of returned tensor
            in case :obj:`edge_weight=None`. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Examples:
        >>> edge_index = Tensor([[0, 1, 1, 2],
        ...                            [1, 0, 2, 1]])
        >>> edge_weight = Tensor([1., 2., 2., 4.])

        >>> # No normalization
        >>> lap = get_laplacian(edge_index, edge_weight)

        >>> # Symmetric normalization
        >>> lap_sym = get_laplacian(edge_index, edge_weight,
                                    normalization='sym')

        >>> # Random-walk normalization
        >>> lap_rw = get_laplacian(edge_index, edge_weight, normalization='rw')
    """
    if normalization is not None:
        assert normalization in ["sym", "rw"]  # 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = mint.ones(edge_index.shape[1], dtype=dtype)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    src, dst = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, src, 0, dim_size=num_nodes, reduce="sum")

    if normalization is None:
        # L = D - A.
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = mint.cat([-edge_weight, deg], dim=0)
    elif normalization == "sym":
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg**-0.5
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        edge_weight = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]

        # L = I - A_norm.
        assert isinstance(edge_weight, Tensor)
        edge_index, edge_weight = add_self_loops(  #
            edge_index, -edge_weight, fill_value=1.0, num_nodes=num_nodes
        )
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float("inf")] = 0
        edge_weight = deg_inv[src] * edge_weight

        # L = I - A_norm.
        assert isinstance(edge_weight, Tensor)
        edge_index, edge_weight = add_self_loops(  #
            edge_index, -edge_weight, fill_value=1.0, num_nodes=num_nodes
        )

    return edge_index, edge_weight


def get_mesh_laplacian(
    crd: Tensor,
    face: Tensor,
    normalization: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the mesh Laplacian of a mesh given by :obj:`pos` and
    :obj:`face`.

    Computation is based on the cotangent matrix defined as

    .. math::
        \mathbf{C}_{ij} = \begin{cases}
            \frac{\cot \angle_{ikj}~+\cot \angle_{ilj}}{2} &
            \text{if } i, j \text{ is an edge} \\
            -\sum_{j \in N(i)}{C_{ij}} &
            \text{if } i \text{ is in the diagonal} \\
            0 & \text{otherwise}
      \end{cases}

    Normalization depends on the mass matrix defined as

    .. math::
        \mathbf{M}_{ij} = \begin{cases}
            a(i) & \text{if } i \text{ is in the diagonal} \\
            0 & \text{otherwise}
      \end{cases}

    where :math:`a(i)` is obtained by joining the barycenters of the
    triangles around vertex :math:`i`.

    Args:
        crd (Tensor): The node positions.
        face (LongTensor): The face indices.
        normalization (str, optional): The normalization scheme for the mesh
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{C}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{M}^{-1/2} \mathbf{C}\mathbf{M}^{-1/2}`

            3. :obj:`"rw"`: Row-wise normalization
            :math:`\mathbf{L} = \mathbf{M}^{-1} \mathbf{C}`
    """
    assert crd.shape[1] == 3 and face.shape[0] == 3

    num_nodes = crd.shape[0]

    def get_cots(left: Tensor, centre: Tensor, right: Tensor) -> Tensor:
        left_pos, central_pos, right_pos = crd[left], crd[centre], crd[right]
        left_vec = left_pos - central_pos
        right_vec = right_pos - central_pos
        dot = Ncon([[-1, 1], [-1, 1]])([left_vec, right_vec])
        cross = ms.numpy.norm(ms.numpy.cross(left_vec, right_vec, axis=1), axis=1)
        cot = dot / cross  # cot = cos / sin
        return cot / 2.0  # by definition

    # For each triangle face, get all three cotangents:
    cot_021 = get_cots(face[0], face[2], face[1])
    cot_102 = get_cots(face[1], face[0], face[2])
    cot_012 = get_cots(face[0], face[1], face[2])
    cot_weight = mint.cat([cot_021, cot_102, cot_012])

    # Face to edge:
    cot_index = mint.cat([face[:2], face[1:], face[::2]], dim=1)
    cot_index, cot_weight = to_undirected(cot_index, cot_weight)

    # Compute the diagonal part:
    cot_deg = scatter(cot_weight, cot_index[0], 0, num_nodes, reduce="sum")
    edge_index, _ = add_self_loops(cot_index, num_nodes=num_nodes)
    edge_weight = mint.cat([cot_weight, -cot_deg], dim=0)

    if normalization is not None:

        def get_areas(left: Tensor, centre: Tensor, right: Tensor) -> Tensor:
            central_pos = crd[centre]
            left_vec = crd[left] - central_pos
            right_vec = crd[right] - central_pos
            cross = ms.numpy.norm(ms.numpy.cross(left_vec, right_vec, axis=1), axis=1)
            area = cross / 6.0  # one-third of a triangle's area is cross / 6.0
            return area / 2.0  # since each corresponding area is counted twice

        # Like before, but here we only need the diagonal (the mass matrix):
        area_021 = get_areas(face[0], face[2], face[1])
        area_102 = get_areas(face[1], face[0], face[2])
        area_012 = get_areas(face[0], face[1], face[2])
        area_weight = mint.cat([area_021, area_102, area_012])
        area_index = mint.cat([face[:2], face[1:], face[::2]], dim=1)
        area_index, area_weight = to_undirected(area_index, area_weight)
        area_deg = scatter(area_weight, area_index[0], 0, num_nodes, "sum")

        if normalization == "sym":
            area_deg_inv_sqrt = area_deg**-0.5
            area_deg_inv_sqrt[area_deg_inv_sqrt == float("inf")] = 0.0
            edge_weight = (
                area_deg_inv_sqrt[edge_index[0]]
                * edge_weight
                * area_deg_inv_sqrt[edge_index[1]]
            )
        elif normalization == "rw":
            area_deg_inv = 1.0 / area_deg
            area_deg_inv[area_deg_inv == float("inf")] = 0.0
            edge_weight = area_deg_inv[edge_index[0]] * edge_weight

    return edge_index, edge_weight
