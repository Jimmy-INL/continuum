import torch
import torchvision
import torchvision.transforms as transforms
from loguru import logger
from torch import nn, optim
import torch.nn.functional as F

# torch.set_grad_enabled(False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def cnn_network():
    net = Net()
    logger.info(net.conv1.weight)


def mulexample():
    net = Net()
    train_set = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
    epochs = range(4)
    running_loss = 0.0
    for epoch in epochs:
        i = 0
        for image, labels in data_loader:
            optimizer.zero_grad()

            pred = net(image)
            # logger.success(pred)
            loss = criterion(pred.float(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            logger.success(loss)
            # if i % 200 == 199:                              # print every 2000 mini-batches
            #     print(
            #         '[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000)
            #     )
            #     running_loss = 0.0
            # logger.warning(loss)
            # logger.success(loss.grad_fn)


if __name__ == "__main__":
    # cnn_network()
    mulexample()
