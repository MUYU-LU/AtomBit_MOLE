from typing import Optional, Tuple, Union

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, mint


def map_index(
    src: Tensor,
    index: Tensor,
    max_index: Optional[Union[int, Tensor]] = None,
    inclusive: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Maps indices in :obj:`src` to the positional value of their
    corresponding occurence in :obj:`index`.
    Indices must be strictly positive.

    Args:
        src (Tensor): The source tensor to map.
        index (Tensor): The index tensor that denotes the new mapping.
        max_index (int, optional): The maximum index value.
            (default :obj:`None`)
        inclusive (bool, optional): If set to :obj:`True`, it is assumed that
            every entry in :obj:`src` has a valid entry in :obj:`index`.
            Can speed-up computation. (default: :obj:`False`)

    :rtype: (:class:`Tensor`, :class:`mindspore.BoolTensor`)

    Examples:
        >>> src = Tensor([2, 0, 1, 0, 3])
        >>> index = Tensor([3, 2, 0, 1])

        >>> map_index(src, index)
        (tensor([1, 2, 3, 2, 0]), tensor([True, True, True, True, True]))

        >>> src = Tensor([2, 0, 1, 0, 3])
        >>> index = Tensor([3, 2, 0])

        >>> map_index(src, index)
        (tensor([1, 2, 2, 0]), tensor([True, True, False, True, True]))

    .. note::

        If inputs are on GPU and :obj:`cudf` is available, consider using RMM
        for significant speed boosts.
        Proceed with caution as RMM may conflict with other allocators or
        fragments.
    """
    if src.is_floating_point():
        raise ValueError(f"Expected 'src' to be an index (got '{src.dtype}')")
    if index.is_floating_point():
        raise ValueError(f"Expected 'index' to be an index (got " f"'{index.dtype}')")
    if max_index is None:
        max_index = max(src.max(), index.max())

    # If the `max_index` is in a reasonable range, we can accelerate this
    # operation by creating a helper vector to perform the mapping.
    # NOTE This will potentially consumes a large chunk of memory
    # (max_index=10 million => ~75MB), so we cap it at a reasonable size:
    THRESHOLD = 10_000_000
    if max_index <= THRESHOLD:
        if inclusive:
            assoc = mint.zeros(max_index + 1, dtype=src.dtype)
        else:
            assoc = -mint.ones(max_index + 1, dtype=src.dtype)
        assoc = ms.ops.scatter_update(assoc, index, mint.arange(index.numel(), dtype=src.dtype))
        out = mint.index_select(assoc, 0, src)

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    import pandas as pd

    left_ser = pd.Series(src.asnumpy(), name="left_ser")
    right_ser = pd.Series(
        index=index.asnumpy(),
        data=pd.RangeIndex(0, index.shape[0]),
        name="right_ser",
    )

    result = pd.merge(
        left_ser, right_ser, how="left", left_on="left_ser", right_index=True
    )

    out_numpy = result["right_ser"].values

    out = Tensor.from_numpy(out_numpy)

    if out.is_floating_point() and inclusive:
        raise ValueError(
            "Found invalid entries in 'src' that do not have "
            "a corresponding entry in 'index'. Set "
            "`inclusive=False` to ignore these entries."
        )

    if out.is_floating_point():
        mask = mint.logical_not(ops.isnan(out))
        out = ops.masked_select(out, mask).astype(index.dtype)
        return out, mask

    if inclusive:
        return out, None
    else:
        mask = out != -1
        return out[mask], mask

def map_index_np(
    src: np.array,
    index: np.array,
    max_index: Optional[Union[int, np.array]] = None,
    inclusive: bool = False,
) -> Tuple[np.array, Optional[np.array]]:
    if src.dtype.kind == 'f':
        raise ValueError(f"Expected 'src' to be an index (got '{src.dtype}')")
    if index.dtype.kind == 'f':
        raise ValueError(f"Expected 'index' to be an index (got " f"'{index.dtype}')")
    if max_index is None:
        max_index = np.max(src.max(), index.max())

    THRESHOLD = 10_000_000
    if max_index <= THRESHOLD:
        if inclusive:
            assoc = np.zeros(max_index + 1, dtype=src.dtype)
        else:
            assoc = -np.ones(max_index + 1, dtype=src.dtype)
        assoc[index] = np.arange(index.size, dtype=src.dtype)
        out = assoc[src]

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    import pandas as pd

    left_ser = pd.Series(src, name="left_ser")
    right_ser = pd.Series(
        index=index,
        data=pd.RangeIndex(0, index.shape[0]),
        name="right_ser",
    )

    result = pd.merge(
        left_ser, right_ser, how="left", left_on="left_ser", right_index=True
    )

    out = result["right_ser"].values

    if out.dtype.kind == 'f' and inclusive:
        raise ValueError(
            "Found invalid entries in 'src' that do not have "
            "a corresponding entry in 'index'. Set "
            "`inclusive=False` to ignore these entries."
        )

    if out.dtype.kind == 'f':
        mask = ~np.isnan(out)
        out = out[mask].astype(index.dtype)
        return out, mask

    if inclusive:
        return out, None
    else:
        mask = out != -1
        return out[mask], mask