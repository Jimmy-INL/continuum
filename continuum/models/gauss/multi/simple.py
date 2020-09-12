import gpytorch
import torch
from gpytorch import variational, kernels, means
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution, VariationalStrategy
)


class RandomizedGP(ApproximateGP):
    """Randomized GP

    Args:
        ApproximateGP ([type]): Initializes the approximate gaussian process with random informaiton.  
    """
    num_tasks = 4
    num_latents = 3
    latent_dim = -1

    def __init__(self):
        init_inducing = torch.randn(100, 10)
        variational_distribution = CholeskyVariationalDistribution(
            init_inducing.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            init_inducing,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

    @property
    def is_trained(self) -> bool:
        return bool(
            int(self.variational_strategy.variational_params_initialized)
        )

    def set_variational_model(self, init_inducing: torch.Tensor):
        self.num_latents = init_inducing.size(0)
        variational_distribution = CholeskyVariationalDistribution(
            init_inducing.size(-2), batch_shape=[self.num_latents]
        )
        self.variational_strategy = variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                init_inducing,
                variational_distribution,
                learn_inducing_locations=True
            ),
            num_tasks=self.num_tasks,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim
        )
        batch_shape = torch.Size([self.num_latents])
        self.mean_module = means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = kernels.ScaleKernel(
            kernels.RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape
        )


class SimpleMultiTaskVariationalModel(RandomizedGP):
    """Simple Multi-Task Variational Inference Model"""

    def __init__(
        self, num_tasks=4, num_latents=3, latent_dim=-1, num_classes: int = 16
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.latent_dim = latent_dim
        # The dimensions after the first are the dimensions worth noting
        self.set_variational_model(torch.rand(num_latents, num_classes))

    def update_var(self, x):
        """x = torch.unsqueeze(x, -1)
        if not self.is_trained:
            self.set_variational_model(x)"""

    def forward(self, x):
        # x = torch.unsqueeze(x, -1)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
