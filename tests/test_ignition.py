#!/usr/bin/env python

from torch import nn
from torch.optim import SGD

from ignite.metrics import Accuracy

from torch_ignition import TorchIgnite
from .network import Net


def test_ignite():
    model = Net()
    optim = SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.NLLLoss()
    t = TorchIgnite(
        model=model,
        optimizer=optim,
        criterion=criterion,
        dataloader=None,
        metric=Accuracy(),
    )

    assert t.evaluator
    assert t.trainer
    assert t.writer
    assert t.model == model
    assert t.criterion == criterion
    assert t.optimizer == optim
