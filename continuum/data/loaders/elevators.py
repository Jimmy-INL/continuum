import os
import urllib.request
from math import floor
import gpytorch
import math
import gpytorch
import torch
import tqdm
from loguru import logger
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import AddedLossTerm, DeepApproximateMLL, VariationalELBO
from gpytorch.models import GP, ApproximateGP
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from scipy.io import loadmat
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset
# from continuum.hidden import DeepGPHiddenLayer


class ElevatorLoader:
    def __init__(self):
        self.smoke_test = ('CI' in os.environ)

    def params(self):
        data = torch.Tensor(loadmat('../elevators.mat')['data'])
        X = data[:, :-1]
        X = X - X.min(0)[0]
        X = 2 * (X / X.max(0)[0]) - 1
        y = data[:, -1]
        return X, y

    def split_data(self, X, y):
        train_n = int(floor(0.8 * len(X)))
        train_x = X[:train_n, :].contiguous()
        train_y = y[:train_n].contiguous()

        test_x = X[train_n:, :].contiguous()
        test_y = y[train_n:].contiguous()
        if torch.cuda.is_available():
            train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(
            ), test_x.cuda(), test_y.cuda()
        return train_x, train_y, test_x, test_y

    @logger.catch(reraise=True)
    def load_data(self, train_x, train_y, test_x, test_y, batch_size=1024):

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader, train_x.shape

    def get_data(self):
        X, y = self.params()
        train_x, train_y, test_x, test_y = self.split_data(X, y)
        return self.load_data(train_x, train_y, test_x, test_y)


# if smoke_test:  # this is for running the notebook in our testing framework
#     X, y = torch.randn(1000, 3), torch.randn(1000)
# else:
#     data = torch.Tensor(loadmat('../elevators.mat')['data'])
#     X = data[:, :-1]
#     X = X - X.min(0)[0]
#     X = 2 * (X / X.max(0)[0]) - 1
#     y = data[:, -1]

# train_n = int(floor(0.8 * len(X)))
# train_x = X[:train_n, :].contiguous()
# train_y = y[:train_n].contiguous()

# test_x = X[train_n:, :].contiguous()
# test_y = y[train_n:].contiguous()

# if torch.cuda.is_available():
#     train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

# train_dataset = TensorDataset(train_x, train_y)
# train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
