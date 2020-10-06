from collections import OrderedDict
from typing import Tuple, Union, List
import creme
from loguru import logger
import numpy as np
import pandas as pd
from continuum.data.generator import make_sarsa_frame


class Windowing:
    """
        Roll through a dataframe manually.
    """
    def __init__(
        self,
        frame: pd.DataFrame,
        window_size: int = 10,
        adaptive_window: bool = False,
        adapted_window_size: int = 1,
        x_label: List[str] = ["state"],
        y_label: List[str] = ["reward"]
    ):
        self.x_label: List[str] = x_label
        self.y_label: List[str] = y_label
        self._current_step: int = 0
        self._window_size: int = window_size
        self._lower_range: int = 0
        self._upper_range: int = len(frame)
        self._frame: pd.DataFrame = frame

        self._is_adapted = adaptive_window
        self._adapted_window_size = adapted_window_size

    @property
    def columns(self) -> list:
        """ Get Columns
            ---
            Return a list of columns from the dataframe.
        """
        return list(self.frame.columns)

    @property
    def window(self):
        """Get Window Size

        Returns:
            int -- The window size
        """
        if self.is_adaptive and self._current_step > self._window_size:
            return self.adapted_window
        return self._window_size

    @property
    def adapted_window(self) -> int:
        """Get the adaptive window size

        Returns:
            int -- Get window size
        """
        return self._adapted_window_size

    @property
    def is_adaptive(self) -> bool:
        return self._is_adapted

    @property
    def frame(self) -> pd.DataFrame:
        """Get the dataframe we're iterating through.

        Returns:
            pd.DataFrame -- The current dataframe
        """
        return self._frame

    @property
    def lower_bounds(self):
        return max((self._current_step - self.window, 0))

    @property
    def upper_bounds(self):
        return min(self._current_step + 1, len(self.frame))

    @property
    def has_next_observation(self) -> bool:
        """Returns if the dataframe has another observation

        Returns
        -------
        bool
            True if there is another observation.
        """
        return self._current_step < self._upper_range - self.window - 1

    @property
    def next_observation(self) -> Union[np.ndarray, pd.DataFrame]:
        return self.frame[self.lower_bounds:self.upper_bounds]

    def step(self) -> pd.DataFrame:
        if not self.has_next_observation:
            raise IndexError("Why are you on the wrong index?")
        self._current_step += 1
        return self.next_observation

    def step_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        frame: pd.DataFrame = self.step()
        x_axis = frame[self.x_label]
        y_axis = frame[self.y_label]
        logger.info((x_axis.head(), y_axis.head()))
        return x_axis, y_axis

    def reset(self):
        self._current_step = 0

    def __len__(self):
        return self._upper_range

    def __iter__(self):
        self._current_step = 0
        while self.has_next_observation:
            x_axis, y_axis = self.step_split()
            yield x_axis, y_axis

    # def __next__(self):
    #     if self.n <= self.count:
    #         self.n += 1
    #         zipped = zip(self.copy())
    #         return next(zipped)
    #     raise StopIteration

    # def __getitem__(self, idx):
    #     if self.zip_list is None:
    #         self.zip_list = list(zip(self.copy()))
    #     return self.zip_list[idx]


if __name__ == "__main__":
    frame = make_sarsa_frame(n_samples=200)
    slider = Windowing(frame)
    for X, y in slider:
        logger.success((X, y))
