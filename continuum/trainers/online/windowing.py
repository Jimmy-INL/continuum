from collections import OrderedDict
from typing import Callable, Dict, List, Union

import creme
import numpy as np
import pandas as pd
from creme.base.estimator import Estimator


class Windowing:
    """Roll through a dataframe manually
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        window_size: int = 10,
        adaptive_window: bool = False,
        adapted_window_size: int = 1,
    ):
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
        return self._current_step < len(self.frame) - self.window - 1

    @property
    def next_observation(self) -> Union[np.ndarray, pd.DataFrame]:
        obs = self.frame[self.lower_bounds:self.upper_bounds]
        return obs

    def step(self):
        if self.has_next_observation:
            self._current_step += 1
            return self.next_observation
        raise IndexError("Why are you on the wrong index?")

    def reset(self):
        self._current_step = 0


if __name__ == "__main__":
    print("Hello World")
