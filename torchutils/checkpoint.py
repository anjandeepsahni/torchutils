import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

class Checkpoint():
    def __init__(self, epoch, model_path, model, optimizer=None, scheduler=None):
        self.epoch = epoch
        self.model_path = model_path
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._state = {'epoch': None,
                        'model': None,
                        'optimizer': None,
                        'scheduler': None}

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, val):
        if val<0:
            raise ValueError('Epoch value must be positive, but got value: {}'.format(val))
        if not isinstance(val, int):
            raise TypeError('Epoch number must be int, but got {}.'.format(type(val)))
        self._epoch = val

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, val):
        if not isinstance(val, str):
            raise TypeError('Model path must be str, but got {}.'.format(type(val)))
        self._model_path = val

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        if val and not isinstance(val, nn.Module):
            raise TypeError('Model must be nn.Module, but got {}.'.format(type(val)))
        self._model = val

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        if val and not isinstance(val, optim.Optimizer):
            raise TypeError('Optimizer must be optim.Optimizer, but got {}.'.format(type(val)))
        self._optimizer = val

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, val):
        if val and not isinstance(val, optim.lr_scheduler._LRScheduler) and \
            not isinstance(val, optim.lr_scheduler.ReduceLROnPlateau):
            raise TypeError('Scheduler must be optim.lr_scheduler._LRScheduler, but got {}.'.format(type(val)))
        self._scheduler = val

    def save(self, metric=0):
        self._state['epoch'] = self.epoch
        if self.model: self._state['model'] = self.model.state_dict()
        if self.optimizer: self._state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler: self._state['scheduler'] = self.scheduler.state_dict()
        metric_str = '%.4f'%(metric)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        model_path = os.path.join(self.model_path,
                    'model_{}_e{}_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), \
                    (str(self.epoch)), metric_str))
        torch.save(self._state, model_path)

    def load(self, ckpt, device=None):
        ckpt_path = os.path.join(self.model_path, ckpt)
        if not os.path.exists(ckpt_path):
            raise ValueError('Checkpoint file does not exist: {}'.format(ckpt_path))
        if device:
            ckpt_dict = torch.load(ckpt_path, map_location=device)
        else:
            ckpt_dict = torch.load(ckpt_path)
        start_epoch = ckpt_dict['epoch'] + 1
        if self.model and ckpt_dict['model']:
            self.model.load_state_dict(ckpt_dict['model'])
        if self.optimizer and ckpt_dict['optimizer']:
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
        if self.scheduler and ckpt_dict['scheduler']:
            self.scheduler.load_state_dict(ckpt_dict['scheduler'])
        return start_epoch, self.model, self.optimizer, self.scheduler

def save_checkpoint(epoch, model_path, model, optimizer=None, scheduler=None, metric=0):
    ckpt = Checkpoint(epoch=epoch, model_path=model_path, model=model, optimizer=optimizer,scheduler=scheduler)
    ckpt.save(metric=metric)

def load_checkpoint(model_path, ckpt_name, model, optimizer=None, scheduler=None, device=None):
    ckpt = Checkpoint(epoch=0, model_path=model_path, model=model, optimizer=optimizer,scheduler=scheduler)
    return ckpt.load(ckpt_name, device)
