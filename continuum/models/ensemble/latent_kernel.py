# Your final neural network will be done from other parts of the project.
import gpytorch
from loguru import logger
import torch
from gpytorch import variational, means, likelihoods

from continuum.models.features.light import DenseLightFeatureExtractor
from continuum.models.gauss.multi.simple import SimpleMultiTaskVariationalModel


class DeepKernelMultiTaskGaussian(gpytorch.Module):
    def __init__(
        self,
        num_tasks: int = 4,
        num_latents: int = 3,
        latent_dim: int = -1,
        num_classes: int = 1000,
        in_channels: int = 4,
        debug: bool = False
    ):
        super(DeepKernelMultiTaskGaussian, self).__init__()
        self.feature_extractor = DenseLightFeatureExtractor(
            in_channels=in_channels, num_classes=num_classes, debug=debug
        )

        # induct_points = num_features
        self.gp_layer = SimpleMultiTaskVariationalModel(
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_classes=num_classes
        )
        # self.num_dim = num_dim

    @property
    def features_num(self) -> int:
        """ Number of Features """
        return self.feature_extractor.out_num()

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp_layer(features)


if __name__ == "__main__":
    # NOTE: The number of tasks must match the number of batches you're entering into the system.
    # BATCH_SIZE=TASK_SIZE
    # WIll have to fill everything inside of the flask size
    num_tasks = 10
    test_shape_x = torch.randn(7, 4, 5, 5)
    test_shape_y = torch.randn(num_tasks)
    logger.debug(test_shape_y)
    dt_model = DeepKernelMultiTaskGaussian(
        num_tasks=num_tasks,
        num_classes=500,
    )

    response = dt_model(test_shape_x)

    likelihood = likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    optimizer = torch.optim.Adam(
        [
            {
                'params': dt_model.gp_layer.parameters()
            },
            {
                'params': likelihood.parameters()
            },
        ],
        lr=0.01,
    )

    dt_model.train()
    likelihood.train()
    # We would have to set a new loss object each time.
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, dt_model.gp_layer, num_data=num_tasks
    )

    optimizer.zero_grad()
    output = dt_model(test_shape_x)
    loss = -mll(output, test_shape_y)
    logger.error(output.rsample().size())
    logger.success(loss)
    loss.backward()
    optimizer.step()

    new_shape_x = torch.randn(2, 4, 5, 5)

    output = dt_model(new_shape_x)
