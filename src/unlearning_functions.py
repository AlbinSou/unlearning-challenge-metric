#!/usr/bin/env python3

import torch.nn as nn
import torch.optim as optim

import src.config as CONFIG
from src.utils import evaluate


def finetune(
    net,
    loader,
    test_loader=None,
    epochs=10,
    weight_decay=5e-4,
    lr=0.01,
    momentum=0.9,
):
    """Simple unlearning by finetuning."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    net.train()

    for ep in range(epochs):
        net.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(CONFIG.DEVICE), targets.to(CONFIG.DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if test_loader is not None:
            print(evaluate(test_loader, net))

    net.eval()


def unlearn_1(
    net,
    retain_loader,
    forget_loader,
):
    finetune(net, retain_loader, lr=0.002, epochs=1)


def unlearn_2(
    net,
    retain_loader,
    forget_loader,
):
    # TODO
    # Insert here your custom unlearning method
    net.linear.reset_parameters()
    finetune(net, retain_loader, lr=0.002, epochs=1)
    return
