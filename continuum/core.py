# import abc
from typing import Any, Dict, List, Optional, Union

import gpytorch
import pandas as pd
import torch
import uuid

# import torch.nn.functional as F
from einops import rearrange
from gpytorch import likelihoods
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from loguru import logger
from toolz.functoolz import memoize
from torch import optim
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from continuum.types import ModelClasses
from tqdm import tqdm, trange

from continuum import Foundation, TrainParams
from continuum.data.generator import make_sarsa_frame
from continuum.data.loaders import times
from continuum.data import decompose_factor
from continuum.trainers import TorchStandardScaler
from continuum.models import DeepKernelMultiTaskGaussian
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, ExponentialLR

ListNumber = List[Union[float, int]]
DEFAULT_TRAINING_PARAMS = TrainParams()

writer = SummaryWriter(f'runs/Continuum/causal-{uuid.uuid4().hex}')


class BaseTrainerInit(Foundation):
    step: int = 0
    train_params: Optional[TrainParams] = DEFAULT_TRAINING_PARAMS
    metrics: Dict[str, ListNumber] = {}
    debug: bool = True

    def __init__(
        self,
        model=DeepKernelMultiTaskGaussian,
        loss=gpytorch.mlls.VariationalELBO,
        likelihood=likelihoods.MultitaskGaussianLikelihood,
        optimizer=optim.SGD,
        epochs: int = 10,
        num_tasks: int = 7,
        num_classes: int = 500,
        **data
    ) -> None:
        super().__init__(**data)
        self._progress_bar: Optional[tqdm] = None

        self.model_types: ModelClasses = ModelClasses(
            model=model,
            likelihood=likelihood,
            optimizer=optimizer,
            loss=loss,
        )
        self.train_params: TrainParams = TrainParams(
            epochs=epochs, num_classes=num_classes, num_tasks=num_tasks
        )
        self.scalar = TorchStandardScaler()

        # Initialize model
        self.init_model()

    @property
    def epoch_range(self) -> Union[tqdm, range]:
        if self.debug:
            return trange(self.train_params.epochs)
        return range(self.train_params.epochs)

    @property
    def progress(self) -> Optional[tqdm]:
        return self._progress_bar

    @property
    def is_progress(self) -> bool:
        return self.progress is not None

    def update_progress_bar(self, dataset: Dataset) -> tqdm:
        self._progress_bar = tqdm(dataset)
        series_count: int = len(dataset)
        self._progress_bar.reset(total=series_count)
        return self._progress_bar

    def update_progress(self):
        if self.debug and self.is_progress:
            self.progress.update()

    def init_model(self):
        model = self.model_types.model
        likelihood = self.model_types.likelihood
        loss = self.model_types.loss
        optimizer = self.model_types.optimizer

        self.model = model(
            num_tasks=self.train_params.num_tasks,
            num_classes=self.train_params.num_classes
        )
        self.likelihood = likelihood(num_tasks=self.train_params.num_tasks)
        self.loss = loss(
            self.likelihood,
            self.model.gp_layer,
            num_data=self.train_params.num_tasks
        )
        self.optimizer = optimizer(
            [
                {
                    'params': self.model.gp_layer.parameters()
                },
                {
                    'params': self.likelihood.parameters()
                },
            ],
            lr=0.001,
            momentum=0.7,
        )

        self.scheduler = CyclicLR(
            self.optimizer,
            base_lr=0.0001,
            max_lr=0.11,
            step_size_up=5,
            mode="exp_range",
            gamma=0.9
        )

    def set_train(self):
        self.model.train()
        self.likelihood.train()

    def set_eval(self):
        self.model.eval()
        self.likelihood.eval()

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return self.scalar.fit_transform(x)

    def timeseries_dataset(self, frame: pd.DataFrame):
        return times.TimeseriesDataset(
            frame=frame,
            x_label=self.train_params.state_fields,
            y_label=self.train_params.reward_fields,
            window=self.train_params.window,
            window_two=self.train_params.window_two
        )


class BaseTrainerComposites(BaseTrainerInit):
    def scc_decomp(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale(decompose_factor(x))

    def train_step(self, X, y):

        self.optimizer.zero_grad()
        output = self.model(X)
        negative_loss = self.loss(output, y)
        loss = -negative_loss

        writer.add_scalar('Training loss', loss, global_step=self.step)
        self.add_metric("loss", loss.item())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.update_progress()
        self.step += 1

    def add_metric(self, name: str, number: Union[float, int]):
        metrics = self.metrics.get(name, [])

        metrics.append(number)
        self.metrics[name] = metrics

    def predict_step(self, X) -> MultivariateNormal:
        return self.model(X)

    def state_dict(self):

        return dict(
            model=self.model.state_dict(),
            optim=self.optimizer.state_dict(),
            loss=self.loss.state_dict(),
            likelihood=self.likelihood.state_dict()
        )

    def load_state_dict(self, state: Dict[str, Any]):
        self.model.load_state_dict(state['model'])
        self.likelihood.load_state_dict(state['likelihood'])
        self.loss.load_state_dict(state['loss'])
        self.optimizer.load_state_dict(state['optim'])


class BaseTrainer(BaseTrainerComposites):
    def fit(self, frame: pd.DataFrame):
        # Reset all training variables (could just create a reset function)
        self.set_train()
        # self.optimizer.zero_grad()
        series = self.timeseries_dataset(frame)

        for _ in self.epoch_range:
            self.optimizer.zero_grad()
            self.update_progress_bar(series)
            for _x, _y in series:
                X_fact = self.scc_decomp(_x)
                self.train_step(X_fact, _y)

    def predict(self, frame: pd.DataFrame):
        # Reset evaluation
        self.set_eval()
        responses = []
        for _x, _y in self.timeseries_dataset(frame):
            resp = self.predict_step(self.scc_decomp(_x))
            responses.append(resp)

        return responses

    def fit_predict(self, frame: pd.DataFrame):
        # Initialize everything to process the steps.
        self.set_train()
        self.optimizer.zero_grad()
        series = self.timeseries_dataset(frame)
        self.update_progress_bar(series)

        final_x: Optional[torch.Tensor] = None
        num_items = len(series) - 1
        idx: int = 0

        for _ in self.epoch_range:
            for x, y in series:
                arr = self.scc_decomp(x)

                if idx == num_items:
                    final_x = arr
                    idx = 0
                    break

                self.train_step(arr, y)
                idx += 1
        self.set_eval()
        prediction = self.predict_step(self.scc_decomp(final_x))
        return prediction.mean, prediction.stddev


if __name__ == "__main__":
    decompose_factor = decompose_factor(shape_val=10)
    frame = make_sarsa_frame(n_samples=200)
    learner = BaseTrainer(epochs=200, optimizer=optim.SGD)
    single_prediction = learner.fit(frame)
    # import matplotlib.pyplot as plt
    # logger.info(learner.metrics)
