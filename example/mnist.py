#!/usr/bin/env python

import argparse
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from ignite.metrics import Accuracy
from torch_ignition import TorchIgnite


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True),
        batch_size=train_batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False),
        batch_size=val_batch_size,
        shuffle=False,
    )
    return dict(train=train_loader, val=val_loader)


def run(args):
    model = Net()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.NLLLoss()
    dataloader = get_data_loaders(args.batch_size, args.val_batch_size)

    ignite = TorchIgnite(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloader=dataloader,
        metric=Accuracy(),
        name="mnist",
        log_dir=args.log_dir,
        log_interval=1,
    )
    ignite(args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1000,
        help="input batch size for validation (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="tensorboard_logs",
        help="log directory for Tensorboard log output",
    )

    args = parser.parse_args()
    run(args)
