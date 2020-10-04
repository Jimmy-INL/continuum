from typing import List, Union
import abc
import gpytorch
import numpy as np
import pandas as pd
import torch as tc
from gpytorch import likelihoods
# from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.likelihoods import Likelihood, likelihood
from loguru import logger
from pydantic import root_validator
from torch import optim
from torch.nn import Module
from torch.nn.modules.loss import _Loss

from continuum import Foundation
from continuum.data.loaders import times
from continuum.models.ensemble.latent_kernel import DeepKernelMultiTaskGaussian


class BaseTrainer(Foundation):
    num_tasks: int = 10
    num_classes: int = 500
    model: Module
    loss: Union[_Loss, Likelihood]
    likelihood: Likelihood
    optimizer: optim.Optimizer

    @root_validator(pre=True)
    def check_modules(cls, values: dict):
        """Check Modules

        Args:
            values (dict): [description]
        """
        task_nums = values.get("num_tasks", 10)
        num_classes = values.get("num_classes", 500)
        model = values.get("model", DeepKernelMultiTaskGaussian
                           )(num_classes=num_classes, num_tasks=task_nums)
        likelihood = values.get(
            "likelihood",
            likelihoods.MultitaskGaussianLikelihood,
        )(num_tasks=task_nums)
        optimizer = values.get("optimizer", optim.Adam)(
            [
                {
                    'params': model.gp_layer.parameters()
                },
                {
                    'params': likelihood.parameters()
                },
            ],
            lr=0.01,
        )

        loss = values.get("loss"
                          )(likelihood, model.gp_layer, num_data=task_nums)

        values['optimizer'] = optimizer
        values['likelihood'] = likelihood
        values['model'] = model
        values['loss'] = loss
        return values

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

    def train_step(self, X, y):
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = -self.loss(output, y)
        loss.backward()
        self.optimizer.step()

    def predict_step(self, X):
        self.optimizer.zero_grad()
        return self.model(X)


class MultitaskTrainer(BaseTrainer):
    model = DeepKernelMultiTaskGaussian
    loss = gpytorch.mlls.VariationalELBO
    likelihood = likelihoods.MultitaskGaussianLikelihood
    optimizer = optim.Adam


if __name__ == "__main__":
    BaseTrainer()
