from mindspore import Tensor, ops, mint


_segment_max = ops.MultitypeFuncGraph('_segment_max')
_segment_amax = ops.MultitypeFuncGraph('_segment_amax')
_segment_min = ops.MultitypeFuncGraph('_segment_min')
_segment_amin = ops.MultitypeFuncGraph('_segment_amin')

_segment_sum = ops.MultitypeFuncGraph('_segment_sum')
_segment_mean = ops.MultitypeFuncGraph('_segment_mean')
_segment_mul = ops.MultitypeFuncGraph('_segment_mul')


@_segment_max.register('Number', 'Tensor')
def _segment_max(axis: int, src: Tensor):
    shape = list(src.shape)
    if shape[axis] == 0:
        shape[axis] = 1
        return mint.zeros(shape)
    return src.max(axis, keepdims=True)


@_segment_amax.register('Number', 'Tensor')
def _segment_amax(axis: int, src: Tensor):
    shape = list(src.shape)
    if shape[axis] == 0:
        shape[axis] = 1
        return mint.zeros(shape).long()
    return src.argmax(axis, keepdims=True)


@_segment_min.register('Number', 'Tensor')
def _segment_min(axis: int, src: Tensor):
    shape = list(src.shape)
    if shape[axis] == 0:
        shape[axis] = 1
        return mint.zeros(shape, dtype=src.dtype)
    return src.min(axis, keepdims=True)


@_segment_amin.register('Number', 'Tensor')
def _segment_amin(axis: int, src: Tensor):
    shape = list(src.shape)
    if shape[axis] == 0:
        shape[axis] = 1
        return mint.zeros(shape).int()
    return src.argmin(axis, keepdims=True)


@_segment_sum.register('Number', 'Tensor')
def _segment_sum(axis: int, src: Tensor):
    shape = list(src.shape)
    if shape[axis] == 0:
        shape[axis] = 1
        return mint.zeros(shape, dtype=src.dtype)
    return src.sum(axis, keepdims=True)


@_segment_mean.register('Number', 'Tensor')
def _segment_mean(axis: int, src: Tensor):
    shape = list(src.shape)
    if shape[axis] == 0:
        shape[axis] = 1
        return mint.zeros(shape, dtype=src.dtype)
    return src.mean(axis, keep_dims=True)


@_segment_mul.register('Number', 'Tensor')
def _segment_mul(axis: int, src: Tensor):
    shape = list(src.shape)
    if shape[axis] == 0:
        shape[axis] = 1
        return mint.zeros(shape, dtype=src.dtype)
    return src.prod(axis, keep_dims=True)


def segment(src: Tensor, ptr: Tensor, dim=0, dim_size=None, reduce: str = "sum") -> Tensor:
    r"""Reduces all values in the first dimension of the :obj:`src` tensor
    within the ranges specified in the :obj:`ptr`. :obj:`mindspore_scatter` package for more
    information.

    Args:
        src (Tensor): The source tensor.
        ptr (Tensor): A monotonically increasing pointer tensor that
            refers to the boundaries of segments such that :obj:`ptr[0] = 0`
            and :obj:`ptr[-1] = src.shape[0]`.
        reduce (str, optional): The reduce operation (:obj:`"sum"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"sum"`)
    """
    vals = mint.split(src, ptr.diff().tolist(), dim=dim)
    common_map = ops.Map()
    if reduce == "max":
        out = common_map(ops.partial(_segment_max, dim), vals)
    elif reduce == "min":
        out = common_map(ops.partial(_segment_min, dim), vals)
    elif reduce == "amax":
        out = common_map(ops.partial(_segment_amax, dim), vals)
    elif reduce == "amin":
        out = common_map(ops.partial(_segment_amin, dim), vals)
    elif reduce == "mul":
        out = common_map(ops.partial(_segment_mul, dim), vals)
    elif reduce in ["sum", "add"]:
        out = common_map(ops.partial(_segment_sum, dim), vals)
    elif reduce == "mean":
        out = common_map(ops.partial(_segment_mean, dim), vals)
    else:
        raise ValueError(f'The value of reduce `{reduce}` is not supported!')

    if dim_size is not None:
        shape = list(src.shape)
        shape[dim] = dim_size - len(vals)
        out += (mint.zeros(shape, dtype=out[0].dtype), )
    val = mint.cat(out, dim=dim)
    return val
