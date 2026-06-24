import math
from typing import Literal, Optional

import mindspore as ms
from mindspore import Tensor, ops, mint


def get_smld_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    num_scales: int,
    dtype: Optional[ms.Type] = None,
) -> Tensor:
    r"""Generates a set of noise values on a logarithmic scale for "Score
    Matching with Langevin Dynamics" from the `"Generative Modeling by
    Estimating Gradients of the Data Distribution"
    <https://arxiv.org/abs/1907.05600>`_ paper.

    This function returns a vector of sigma values that define the schedule of
    noise levels used during Score Matching with Langevin Dynamics.
    The sigma values are determined on a logarithmic scale from
    :obj:`sigma_max` to :obj:`sigma_min`, inclusive.

    Args:
        sigma_min (float): The minimum value of sigma, corresponding to the
            lowest noise level.
        sigma_max (float): The maximum value of sigma, corresponding to the
            highest noise level.
        num_scales (int): The number of sigma values to generate, defining the
            granularity of the noise schedule.
        dtype (ms.Type, optional): The output data type.
            (default: :obj:`None`)
    """
    out = ops.linspace(
        math.log(sigma_max),
        math.log(sigma_min),
        num_scales,
    ).exp()

    if dtype is not None:
        out = out.astype(dtype)
    return out


def get_diffusion_beta_schedule(
    schedule_type: Literal["linear", "quadratic", "constant", "sigmoid"],
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
    dtype: Optional[ms.Type] = None,
) -> Tensor:
    r"""Generates a schedule of beta values according to the specified strategy
    for the diffusion process from the `"Denoising Diffusion Probabilistic
    Models" <https://arxiv.org/abs/2006.11239>`_ paper.

    Beta values are used to scale the noise added during the diffusion process
    in generative models. This function creates an array of beta values
    according to a pre-defined schedule, which can be either :obj:`"linear"`,
    :obj:`"quadratic"`, :obj:`"constant"`, or :obj:`"sigmoid"`.

    Args:
        schedule_type (str): The type of schedule to use for beta values.
        beta_start (float): The starting value of beta.
        beta_end (float): The ending value of beta.
        num_diffusion_timesteps (int): The number of timesteps for the
            diffusion process.
        dtype (ms.Type, optional): The output data type.
            (default: :obj:`None`)
        ops.cat (ops.cat, optional): The output ops.cat.
            (default: :obj:`None`)
    """
    if schedule_type == "linear":
        out = ops.linspace(beta_start, beta_end, num_diffusion_timesteps)

    elif schedule_type == "quadratic":
        out = ops.linspace(
            beta_start**0.5, beta_end**0.5, num_diffusion_timesteps
        ) ** 2
    elif schedule_type == "constant":
        return ops.full((num_diffusion_timesteps,), fill_value=beta_end)

    elif schedule_type == "sigmoid":
        out = ops.linspace(-6, 6, num_diffusion_timesteps).sigmoid() * (beta_end - beta_start) + beta_start
    else:
        raise ValueError(f"Found invalid 'schedule_type' (got '{schedule_type}')")

    if dtype is not None:
        out = out.astype(dtype=dtype)
    return out
