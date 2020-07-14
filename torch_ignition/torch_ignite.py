#!/usr/bin/env python

import os
import os.path as osp
from datetime import datetime
import numpy as np

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ignite.engine import (Events, create_supervised_trainer,
                           create_supervised_evaluator)
from ignite.metrics import Accuracy, Loss
from torch_ignition.logger import get_logger


def get_name(prefix=''):
    return prefix + '_' + datetime.now().strftime("%Y%m%d%H%M%S")


def create_logging_folder(log_dir: str,
                          appname: str = 'train'):
    if not osp.isdir(log_dir):
        os.mkdir(log_dir)
    log_dir = osp.join(log_dir, get_name(appname))
    os.mkdir(log_dir)
    return log_dir


class TorchIgnite(object):

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion,
                 dataloader: dict,
                 **kwargs):
        super(TorchIgnite, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader

        self.log_interval = kwargs.get('log_interval', 1)
        name = kwargs.get('name', self.__class__.__name__)
        device_id = kwargs.get('device_id', 0)

        # logger
        self.logger = get_logger(name)

        # device
        self.device = torch.device(
            f'cuda:{device_id}'
            if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'Device: {self.device}')

        # logging directory
        log_dir = kwargs.get('log_dir', '../logs')
        self.log_dir = create_logging_folder(log_dir, name.lower())
        self.logger.info(f'Data will be logged at {self.log_dir}')

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.model.to(self.device)
        self.criterion.to(self.device)

        self.trainer = create_supervised_trainer(
            self.model, self.optimizer, self.criterion,
            device=self.device)

        val_metrics = dict(accuracy=Accuracy(), loss=Loss(criterion))
        self.evaluator = create_supervised_evaluator(
            self.model, metrics=val_metrics, device=self.device)

        self.logger.info('Setup completed')

    def __call__(self, epochs: int):
        self.logger.info('Start ignition')
        self.logger.info(f'Total epochs: {epochs}')

        self.run(epochs=epochs)
        self.writer.close()

    def run(self, epochs: int = 1):
        trainer = self.trainer
        train_loader = self.dataloader['train']
        val_loader = self.dataloader['val']

        @trainer.on(Events.ITERATION_COMPLETED(every=self.log_interval))
        def log_training_loss(engine):
            length = len(train_loader)
            self.logger.info(
                f'Epoch[{engine.state.epoch}] ' +
                f'Iteration[{engine.state.iteration}/{length}] ' +
                f'Loss: {engine.state.output:.2f}')
            self.writer.add_scalar(
                "training/loss", engine.state.output,
                engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.evaluator.run(train_loader)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            self.logger.info(
                f'Training Results - Epoch: {engine.state.epoch}  ' +
                f'Avg accuracy: {avg_accuracy:.2f}' +
                f'Avg loss: {avg_loss:.2f}')
            self.writer.add_scalar(
                'training/avg_loss', avg_loss, engine.state.epoch)
            self.writer.add_scalar(
                'training/avg_accuracy', avg_accuracy,
                engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            self.evaluator.run(val_loader)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            self.logger.info(
                f'Validation Results - Epoch: {engine.state.epoch} ' +
                f'Avg accuracy: {avg_accuracy:.2f} ' +
                f'Avg loss: {avg_loss:.2f}')
            self.writer.add_scalar(
                'valdation/avg_loss', avg_loss, engine.state.epoch)
            self.writer.add_scalar(
                'valdation/avg_accuracy', avg_accuracy,
                engine.state.epoch)

        trainer.run(train_loader, max_epochs=epochs)

    def trainable_parameters(self):
        total_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                total_params += np.prod(parameter.size())
        self.logger.info(f'total_params: {total_params}')

    def __str__(self):
        return str(self.__call__.__name__.upper())

    def __del__(self):
        try:
            del self.writer
            del self.self.evaluator
            del self.trainer
            del self.dataloader
        except BaseException:
            pass
