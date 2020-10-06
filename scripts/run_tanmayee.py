import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import gpytorch
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import pytest
import ta
from loguru import logger

from darwin_ml import __version__
from darwin_ml.technical import (
    fibonacci, fibonacci_rsi, super_hyper_mega_average_true_range
)
from darwin_ml.technical.momentum import rsi_positions
from darwin_ml.technical.signals import fib_intensity_signal
from darwin_ml.technical.volume import fibonacci_boll_bands
from darwin_ml.utils import Windowing
from darwin_ml.utils.preprocessing import (
    format_look_ahead, format_timeseries_dataframe
)
from typing import List
from copy import copy, deepcopy
from creme import stats
from creme import datasets
from creme import linear_model
from creme import metrics
from creme import preprocessing

from torch import optim
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.variational import AdditiveGridInterpolationVariationalStrategy, CholeskyVariationalDistribution


def boolean_flip(item):
    if item == True:
        return 1
    else:
        return -1


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def stock_data():
    BASE_DIR = Path(__file__
                    ).resolve().parent.parent.cwd() / 'data' / 'stock_data.csv'
    BASE_DIR_STR = str(BASE_DIR)
    return pd.read_csv(BASE_DIR_STR)


def main():
    df = stock_data()
    df = ta.utils.dropna(df)
    df = format_timeseries_dataframe(df, "Timestamp")
    df = format_look_ahead(df, "Close", size=-4)
    df.dropna()
    df['log_returns'] = 0
    df['log_returns'] = np.where(df["Close_future"] > df["Close"], 1, 1)
    df['log_returns'] = np.where(
        df["Close_future"] < df["Close"], -1, df['log_returns']
    )
    df = fibonacci(df)
    df = fibonacci_rsi(df)
    # df = super_hyper_mega_average_true_range(df)
    df = df.drop(
        columns=[
            'Open', 'High', 'Low', 'Volume_Currency', 'Weighted_Price',
            'Volume_BTC', 'Close', 'above_below_close', 'Close_future'
        ]
    )
    df = df.rename(columns={"log_returns": "y"})
    return df


def roll_dataframe_stats(
    frame: pd.DataFrame,
    window=14,
    min_steps: int = 1,
    callback: Optional[Callable] = None,
    metric: metrics.ClassificationReport = metrics.ClassificationReport()
):
    windower = Windowing(
        frame,
        window_size=window,
        adaptive_window=False,
        adapted_window_size=0
    )

    # while windower.has_next_observation:
    #     res = windower.step()
    #     x = res.to_dict(orient="record")[0]
    #     y = x.pop("y")

    #     if model_copy is not None:
    #         y_pred = boolean_flip(model.predict_one(x))
    #         model.fit_one(x, y)
    #         if y_pred != y:
    #             prob_up = model.predict_proba_one(x)
    #             prob_values = list(prob_up.values())
    #    window=14,
    #                      min_steps: int = 1,
    #                      callback: Optional[Callable] = None,
    #                      metric: metrics.ClassificationReport = metrics.ClassificationReport()):

    step_count = 0
    history = []

    model = None
    model_copy = copy(model)

    _mean_down = stats.Mean()
    _mean_up = stats.Mean()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    has_train = False
    model = ExactGPModel(torch.tensor([[]]), torch.tensor([]), likelihood)
    optimizer = torch.optim.Adam(
        [
                                             # Includes GaussianLikelihood parameters
            {
                'params': model.parameters()
            },
        ],
        lr=0.1
    )
    prior_training = None
    while windower.has_next_observation:
        if not windower.is_between_bounds:
            windower.step(incr_only=True)
            continue
        res = windower.step()
        train_x = torch.tensor(res.values.astype(np.float32))
        y = res.pop("y")
        train_y = torch.tensor(y.values.astype(np.float32))
        print(train_x.size())
        print(train_y.size())
        if has_train is False:
            model = ExactGPModel(train_x, train_y, likelihood)
            model.train()
            likelihood.train()
            has_train = True
        else:
            model.eval()
            likelihood.eval()
            logger.warning(model)
            predicted = model(train_x)
            model = model.get_fantasy_model(train_x, train_y)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(
            [
                                                 # Includes GaussianLikelihood parameters
                {
                    'params': model.parameters()
                },
            ],
            lr=0.1
        )

        prior_information = train_x
        training_iter = 5
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            loss = -mll(output, train_y)
            loss.backward()
            print(
                'Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                )
            )
            optimizer.step()

    return step_count >= min_steps, history


def test():
    df = main()
    report = metrics.ClassificationReport()
    roll_dataframe_stats(df, metric=report)


test()
