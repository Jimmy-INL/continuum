from dataclasses import dataclass
from typing import Optional, Tuple

import gpytorch
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP
from torch import LongStorage

from continuum.models.gauss.deep import ApproximateDeepGPHiddenLayer


class DeepDoublyGaussianProcess(DeepGP):

    def __init__(self, train_x_shape, num_output_dims=10):
        hidden_layer = ApproximateDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
        )

        last_layer = ApproximateDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output


@dataclass
class LossParams:
    frame_y: str = "reward"
    frame_y_hat: str = "reward_hat"


@dataclass
class TrainParams:
    state_name: str = "state"
    action_name: str = "action"
    reward_name: str = "reward"


class DoublyContainer:
    """DoublyContainer

    Contains Estimator Functions:
    
    * fit
    * predict
    * adjust_loss
    """

    def __init__(
        self,
        loss_params: LossParams = LossParams(),
        train_params: TrainParams = TrainParams()
    ) -> None:
        self.loss_params = loss_params
        self.train_params = train_params
        self._model: Optional[DeepDoublyGaussianProcess] = None

    @property
    def model(self) -> DeepDoublyGaussianProcess:
        if self._model is None:
            raise AttributeError("Model not found")
        return self._model

    @property
    def is_model(self) -> bool:
        return self._model is not None

    def throw_model(self):
        if not self.is_model:
            raise AttributeError("Model not found")

    def activate_model(self, shape):
        self._model = DeepDoublyGaussianProcess(shape)
        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def adjust_loss(self, loss_dataframe: pd.DataFrame) -> bool:
        """Adjust Estimates With Loss

        Takes a dataframe that contains both the estimated reward and the realized reward form the system.

        Parameters
        ----------
        loss_dataframe : pd.DataFrame
            The dataframe that the estimations and the realized rewards.

        Returns
        -------
        bool
            Determines if the dataframe was trained or not.
        """
        try:
            self.throw_model()
        except AttributeError:
            return False
        return True

    def fit(self, train_frame: pd.DataFrame) -> bool:
        """Train The DeepFrame

        Fits the Deep gaussian process network to a given dataframe. Will create network if we don't have one already.

        Parameters
        ----------
        train_frame : pd.DataFrame
            Train Dataframe

        Returns
        -------
        bool
            Returns true if the fit function actually works.
        """
        X, y = np.random.uniform(size=(4, 44)), np.random.uniform(1, 0, 4)
        try:
            self.throw_model()
        except AttributeError:
            x_size = X.shape
            self.activate_model(x_size)
        self.model.train()
        return True

    def predict(
        self, predict_frame: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts Reward

        Predicts the reward given the massive concattenated state space.

        Parameters
        ----------
        predict_frame : pd.DataFrame
            A dataframe has a state space and associated actions to determine a reward.

        Returns
        -------
        tuple
        """
        self.throw_model()
        self.model.eval()
        mean = torch.ones([1, 4], dtype=torch.float64)
        var = torch.ones([1, 4], dtype=torch.float64)

        return (mean, var, mean)

    def get_state(self) -> dict:
        try:
            self.throw_model()
            return self.model.state_dict()
        except AttributeError:
            return {}

    def set_state(self, state_dict_dict: dict):
        self.model.load_state_dict(state_dict_dict)

    # def predict(self, test_loader):
    #     with torch.no_grad():
    #         mus = []
    #         variances = []
    #         lls = []
    #         for x_batch, y_batch in test_loader:
    #             preds = model.likelihood(model(x_batch))
    #             mus.append(preds.mean)
    #             variances.append(preds.variance)
    #             lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

    #     return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


if __name__ == "__main__":
    fit_frame = pd.DataFrame({
        "state": [],
        "action": [],
        "reward": [],
    })
    doubly_container = DoublyContainer()
    doubly_container.fit(fit_frame)
    is_trained = doubly_container.adjust_loss(fit_frame)
    assert is_trained is True, "The model doesn't exist yet."
    X, y, z = doubly_container.predict(fit_frame)
    print((X, y, z))