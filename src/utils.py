#!/usr/bin/env python3
import ray
import torch
import torchvision
import torchvision.transforms as transforms

import src.config as CONFIG

class BlockingScheduler:
    def __init__(self, max_launched):
        self.max_launched = max_launched
        self.scheduled_tasks = []

    def schedule(self, function, *args, **kwargs):
        scheduled_task = None
        if len(self.scheduled_tasks) > self.max_launched:
            ready, _ = ray.wait(self.scheduled_tasks, num_returns=1)
            self.scheduled_tasks.remove(ready[0])
            scheduled_task = self.schedule(function, *args, **kwargs)
        else:
            scheduled_task = function.remote(*args, **kwargs)
            self.scheduled_tasks.append(scheduled_task)
        return scheduled_task


def load_cifar10(use_transforms=True):
    default_cifar10_train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    default_cifar10_eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    if use_transforms:
        train_transform = default_cifar10_train_transform
    else:
        train_transform = default_cifar10_eval_transform

    train = torchvision.datasets.CIFAR10(
        root="/DATA/data", train=True, transform=train_transform
    )
    test = torchvision.datasets.CIFAR10(
        root="/DATA/data", train=False, transform=default_cifar10_eval_transform
    )
    return train, test


@torch.no_grad()
def evaluate(loader, model):
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(CONFIG.DEVICE), y.to(CONFIG.DEVICE)
        out = model(x)
        pred = out.argmax(1)
        total += len(x)
        correct += (pred == y).float().sum()

    return correct / total
