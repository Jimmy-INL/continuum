import torch.nn as nn
from loguru import logger
from torchvision.models import MNASNet


class DenseLightFeatureExtractor(MNASNet):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        debug: bool = True,
        **kwargs
    ) -> None:
        super(DenseLightFeatureExtractor,
                self).__init__(alpha=1, num_classes=num_classes, **kwargs)
        self.is_debug = debug
        self.in_channels = in_channels
        self.update_first()
        if self.is_debug:
            self.print_variables()

    def update_first(self):
        """ Update the first convolutional layer to fit the number of channels for the feature extractor. """
        self.layers[0] = nn.Conv2d(
            self.in_channels,
            32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )

    def out_num(self) -> int:
        return self.classifier[1].out_features

    def print_variables(self):
        logger.info("First Convolutional Network")
        logger.debug(self.layers[0])
        logger.info("Output number")
        logger.success(self.out_num())


if __name__ == "__main__":
    DenseLightFeatureExtractor(in_channels=4, num_classes=1500)
