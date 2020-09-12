import copy as cp
import itertools
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import toolz
import torch
import torchvision.transforms as transforms
from loguru import logger
from more_itertools import (
    collapse, random_combination, random_permutation, repeatfunc
)
from toolz import curry
from toolz.curried import (compose, map, pipe, random_sample, sliding_window)
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from continuum.data.generator import make_sarsa_frame

settings = {"handlers": [{"sink": sys.stdout, "enqueue": True}]}

logger.configure(**settings)

currnet_folder = Path(__file__).parent
to_tensor = transforms.ToTensor()


def rainbow_log():
    return random.choice([
        logger.info, logger.debug, logger.success, logger.error
    ])


def compress_row(x):
    return np.hstack(zip(x))


def concat_tuple(x):
    return list(toolz.concat(x))


def log_el(x):
    return x


def stack_process(x):
    tup = tuple(torch.tensor(z) for z in x)
    return torch.stack(tup).unsqueeze(0)


@curry
def stack_array(size, x):
    if size <= 0: return stack_process(x)
    slide_size = sliding_window((size + 1))
    slide_stack = map(stack_process, slide_size(x))
    tuple_stack = tuple(slide_stack)
    return (torch.stack(tuple_stack, dim=1)).squeeze(0)


@curry
def slice_target(size1, size2, x):
    if size2 in [0, -1]: return x[-1]
    diff = -(size1 - size2)
    return x[diff:]


@curry
def random_window(window_range, window):
    return sliding_window(random.randint(*window_range), window)


def random_split_windows(
    window, start: int = 3, end: int = 10, select_prob: float = 0.7
):
    set_range = random_window((start, end))
    yield from random_permutation(
        random_sample(select_prob, set_range(window))
    )


def iter_random_window_split(
    zipped_window,
    is_copied: bool = True,
    start: int = 3,
    end: int = 10,
    select_prob: float = 0.7
):
    elements = zipped_window if is_copied else cp.deepcopy(zipped_window)
    all_windows = random_split_windows(
        elements, start=start, end=end, select_prob=select_prob
    )
    # logger.info(next(all_windows))
    # collapse(place, levels=1)
    yield from itertools.chain(all_windows)


class TimeseriesDataset(Dataset):
    """Loads a TimeSeries Dataset"""
    def __init__(
        self,
        frame: pd.DataFrame,
        window: int = 10,
        window_two: int = 3,
        x_label: List[str] = ["state"],
        y_label: List[str] = ["reward"]
    ):
        self.n = 0
        local_frame = frame.copy()
        np_conv = lambda x: x.to_numpy()
        swindows = sliding_window(window)

        func = lambda x: pipe(x, np_conv, swindows)
        stay = stack_array(window_two)
        _slice_target = slice_target(window, window_two)
        self.x_axis = pipe(
            local_frame[x_label], np_conv, map(compress_row), swindows,
            map(compose(concat_tuple)), map(stay)
        )
        self.y_axis = pipe(
            frame[y_label], func, map(lambda x: torch.tensor(x).view(-1)),
            map(log_el), map(_slice_target)
        )

        self.count = toolz.count(self.copy()[0])

        self.zip_list = None

    def copy(self):
        return cp.deepcopy(self.x_axis), cp.deepcopy(self.y_axis)

    def zipped_copy(self):
        return zip(*(self.copy()))

    def random_fold(self):
        folded = iter_random_window_split(self.zipped_copy())
        return collapse(folded, levels=1)

    def first(self, index: int = 0):
        copied = self.copy()
        return toolz.first(copied[index])

    def x_shape(self):
        return self.first().shape

    def y_shape(self):
        return self.first(1).shape

    def __len__(self):
        return self.count

    def __iter__(self):
        self.n = 0
        for x, y in self.random_fold():
            self.n += 1
            yield x, y

    def __next__(self):
        if self.n <= self.count:
            self.n += 1
            zipped = zip(self.copy())
            return next(zipped)
        raise StopIteration

    def __getitem__(self, idx):
        if self.zip_list is None:
            self.zip_list = list(zip(self.copy()))
        return self.zip_list[idx]


if __name__ == "__main__":
    epochs = 5
    frame = make_sarsa_frame(n_samples=200)
    for _ in range(epochs):
        time_series = TimeseriesDataset(frame, x_label=["state", "actions"])
        for x, y in time_series:
            logger.warning((x.shape, y.shape))
            break
        break
    #         rain_log = rainbow_log()
    #         rain_log("ANOTHER ONE")
    #         # break
    # logger.complete()
    # logger.info("done")
