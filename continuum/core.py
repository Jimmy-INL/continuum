import abc
from typing import Any, Dict, List, Optional, Type, Union

import gpytorch
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from gpytorch import likelihoods
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from loguru import logger
from pydantic import BaseModel
from toolz.functoolz import memoize
from torch import optim
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from continuum import ExtraResp
from continuum.data.generator import make_sarsa_frame
from continuum.data.loaders import times
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


SHAPE_VALUE = 10


def decompose_factor(x_arr: torch.Tensor) -> torch.Tensor:
    x_shape = x_arr.shape
    dividing_vals = get_sorted_fact(x_shape)
    new_shape = rearrange(
        x_arr, 'x y (b1 b2) -> x y b1 b2', b1=dividing_vals[0]
    )
    new_shape = F.interpolate(new_shape, (SHAPE_VALUE, SHAPE_VALUE))
    new_shape = new_shape.float()
    return new_shape


ListNumber = List[Union[float, int]]


class BaseTrainer:
    epochs: int = 10
    num_tasks: int = 7
    num_classes: int = 500
    model: Optional[Module]
    likelihood: Optional[Likelihood]
    optimizer: Optional[Optimizer]
    loss: Optional[Module]
    model_types = Optional[Dict[str, Union[Type]]]
    metrics: Dict[str, ListNumber] = {}

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def __init__(
        self,
        extra_resp: ExtraResp = ExtraResp.IGNORE,
        model=DeepKernelMultiTaskGaussian,
        loss=gpytorch.mlls.VariationalELBO,
        likelihood=likelihoods.MultitaskGaussianLikelihood,
        optimizer=optim.SGD,
        **data
    ) -> None:
        self.Config.extra = extra_resp.value
        super().__init__(**data)
        self.model_types = dict(
            model=model, likelihood=likelihood, optimizer=optimizer, loss=loss
        )
        self.init_model()

    def init_model(self):
        model = self.model_types['model']
        likelihood = self.model_types['likelihood']
        loss = self.model_types['loss']
        optimizer = self.model_types['optimizer']

        self.model = model(
            num_tasks=self.num_tasks, num_classes=self.num_classes
        )
        self.likelihood = likelihood(num_tasks=self.num_tasks)
        self.loss = loss(
            self.likelihood, self.model.gp_layer, num_data=self.num_tasks
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
            lr=0.01,
        )

    def set_train(self):
        self.model.train()
        self.likelihood.train()

    def set_eval(self):
        self.model.eval()
        self.likelihood.eval()

    def fit(
        self,
        frame: pd.DataFrame,
        x_label: List[str] = ["state"],
        y_label: List[str] = ["reward"],
        window: int = 10,
        window_two: int = 3,
    ):
        self.set_train()
        for _x, _y in times.TimeseriesDataset(
            frame=frame,
            x_label=x_label,
            y_label=y_label,
            window=window,
            window_two=window_two
        ):
            self.train_step(_x, _y)

    def predict(
        self,
        frame: pd.DataFrame,
        x_label: List[str] = ["state"],
        y_label: List[str] = ["reward"],
        window: int = 10,
        window_two: int = 3,
    ):
        self.set_eval()
        responses = []
        for _x, _y in times.TimeseriesDataset(
            frame=frame,
            x_label=x_label,
            y_label=y_label,
            window=window,
            window_two=window_two
        ):
            resp = self.predict_step(_x)
            responses.append(resp)

        return responses

    def fit_predict(
        self,
        frame: pd.DataFrame,
        x_label: List[str] = ["state"],
        y_label: List[str] = ["reward"],
        window: int = 10,
        window_two: int = 3,
    ):
        final_x: Optional[torch.Tensor] = None
        items = times.TimeseriesDataset(
            frame=frame,
            x_label=x_label,
            y_label=y_label,
            window=window,
            window_two=window_two
        )
        num_items = len(items) - 1
        idx: int = 0
        self.set_train()
        for _ in range(self.epochs):
            for x, y in items:
                arr = decompose_factor(x)
                if idx == num_items:
                    final_x = arr
                    idx = 0
                    break

                self.train_step(arr, y)
                idx += 1
        self.set_eval()
        prediction = self.predict_step(final_x)
        return prediction.mean, prediction.stddev

    def train_step(self, X, y):

        self.optimizer.zero_grad()
        output = self.model(X)
        negative_loss = self.loss(output, y)
        loss = -negative_loss
        self.add_metric("loss", loss.item())
        loss.backward()
        self.optimizer.step()

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


if __name__ == "__main__":
    frame = make_sarsa_frame(n_samples=20)
    learner = BaseTrainer()
    single_prediction = learner.fit_predict(
        frame, x_label=["state", "actions"]
    )
    logger.info(learner.metrics)
