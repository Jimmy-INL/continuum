import torch
import torch.nn.functional as F
from einops import rearrange
from toolz.functoolz import memoize
from toolz import curry

from continuum.models.ensemble.latent_kernel import DeepKernelMultiTaskGaussian


def all_factors(value: int):
    factors = []
    for i in range(1, int(value**0.5) + 1):
        if value % i == 0:
            fact = value / i
            factors.append((i, fact, abs(fact - i)))
    return factors


def is_prime(x: int):
    if x <= 1:
        return False
    return all(x % i != 0 for i in range(2, x))


@memoize
def get_sorted_fact(x_shape) -> int:
    last_item = x_shape[-1]
    if is_prime(last_item):
        raise ValueError("We can't split a prime number. Please change it.")
    last_factors = all_factors(last_item)
    sorted_la_facts = sorted(last_factors, key=lambda x: x[-1])
    return sorted_la_facts[0]


# SHAPE_VALUE = 5


@curry
def decompose_factor(x_arr: torch.Tensor, shape_val: int = 5) -> torch.Tensor:
    x_shape = x_arr.shape
    dividing_vals = get_sorted_fact(x_shape)
    new_shape = rearrange(
        x_arr, 'x y (b1 b2) -> x y b1 b2', b1=dividing_vals[0]
    )
    new_shape = F.interpolate(new_shape, (shape_val, shape_val))
    new_shape = new_shape.float()
    return new_shape