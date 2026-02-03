import functools
import torch

from typing import List, Tuple


@functools.cache
def progressive_checkerboard(n: int) -> torch.LongTensor:
    '''
    Produce the progressive checkerboard scan order for an n x n grid.

    Args:
        n (int): The size of the grid (n x n).

    Returns:
        torch.LongTensor: A tensor of shape (n, n) containing the order indices.
    '''

    assert n > 0
    orig_n = n
    if (n & (n - 1)) != 0:
        # round up to power of 2
        n = 1 << (n - 1).bit_length()

    # generate coords in progressive checkerboard order
    coords = progressive_checkerboard_coords(n, 0, 0)

    # convert coords to rank indices
    order = torch.empty((n, n), dtype=torch.long)
    for rank, (x, y) in enumerate(coords):
        order[y, x] = rank

    if n != orig_n:
        # crop and reassign ranks to sorted order 
        order = order[:orig_n, :orig_n]
        sorted_indices = torch.argsort(order.reshape(-1))
        order = torch.empty_like(order)
        order.reshape(-1)[sorted_indices] = torch.arange(orig_n * orig_n, dtype=torch.long)

    return order


def progressive_checkerboard_coords(size, x, y):
    if size == 1:
        return [(x, y)]

    # divide into quadrants
    d = size // 2

    # create balanced lists for TL, BR, TR, BL
    sublists = [
        progressive_checkerboard_coords(d, xi, yi)
        for xi, yi in [(x, y), (x + d, y + d), (x + d, y), (x, y + d)]
    ]

    # combine round-robin from the quadrants
    return concat(zip(*sublists))


def concat(lists):
    result = []
    for lst in lists:
        result.extend(lst)
    return result
