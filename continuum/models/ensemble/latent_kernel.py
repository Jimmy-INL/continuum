# Your final neural network will be done from other parts of the project.
import gpytorch
import torch
from gpytorch import variational, means, likelihoods

from continuum.models.features.light import DenseLightFeatureExtractor
from continuum.models.gauss.multi.simple import VarGPModel


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
        self.gp_layer = VarGPModel(
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
    text_shape_x = torch.randn(7, 4, 5, 5)
    deep_task = DeepKernelMultiTaskGaussian(
        num_tasks=10,
        num_classes=1500,
    )
    deep_task(text_shape_x)