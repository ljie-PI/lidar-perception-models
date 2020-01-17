#!/usr/bin/env python

import torch

def create_optimizer(model, train_config):
    params = model.parameters()
    if train_config.optimizer.lower() == "adam":
        return torch.optim.Adam(
            params, lr=train_config.learning_rate)
    elif train_config.optimizer.lower() == "adadelta":
        return torch.optim.Adadelta(
            params, lr=train_config.learning_rate)
    elif train_config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            params, lr=train_config.learning_rate)
    else:
        raise Exception("supported optimizer are adam, adadelta and sgd")