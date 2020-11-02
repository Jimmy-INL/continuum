from typing import Optional, Union, Dict, NewType, Type
from pydantic.main import BaseModel
from torch import optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.optimizer import Optimizer
from gpytorch.likelihoods import Likelihood
from gpytorch import likelihoods, mlls, Module as GModule
from continuum.models import DeepKernelMultiTaskGaussian
from continuum.types import AllowedBase
MachineTypes = Union[Optimizer, Likelihood, Module, GModule]
TorchTypes = NewType('NewTorch', MachineTypes)

MutableMachineMap = Dict[str, MachineTypes]
OptionMachine = Optional[Union[MachineTypes]]


class ModelClasses(AllowedBase):
    """
    ModelClasses Define Models

    We use this here to define models inside of a consistent way. It's declared inside of the training system.
    """
    model: Type[MachineTypes] = DeepKernelMultiTaskGaussian
    loss: Type[MachineTypes] = mlls.VariationalELBO
    likelihood: Type[MachineTypes] = likelihoods.MultitaskGaussianLikelihood
    optimizer: Type[MachineTypes] = optim.SGD