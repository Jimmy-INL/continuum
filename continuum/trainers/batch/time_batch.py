import pandas as pd
from loguru import logger
from continuum.data.loaders import times


class TimeBatchTrainer:
    """Time Batch Learning
    Instead of taking one sample we take many samples then learn batches of them for time series.
    """
    def __init__(self, frame: pd.DataFrame) -> None:
        self.times_frame = times.TimeseriesDataset(frame)

    def train(self):
        logger.debug("Training on test data")
