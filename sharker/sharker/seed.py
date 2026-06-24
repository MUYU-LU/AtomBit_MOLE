import random

import numpy as np
import mindspore as ms


def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :mindspore:`Mindspore`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
