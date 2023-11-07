import argparse
import copy
import random

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim

from src.resnet import resnet18
from src.utils import BlockingScheduler, evaluate, load_cifar10
import src.config as CONFIG


def finetune(
    net,
    loader,
    test_loader=None,
    epochs=10,
    weight_decay=5e-4,
    lr=0.01,
    momentum=0.9,
    use_scheduler=True,
):
    """Simple unlearning by finetuning."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
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
        if use_scheduler:
            scheduler.step()

        if test_loader is not None:
            acc = evaluate(test_loader, net)
            print(f"Epoch {ep}, Acc: {float(acc.cpu().numpy())}")

    net.eval()
    return net


@ray.remote(num_gpus=0.2, num_cpus=1)
def train_and_save_checkpoint(indices, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train, test = load_cifar10()
    retain_set = torch.utils.data.Subset(train, indices)

    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.to(CONFIG.DEVICE)

    train_loader = torch.utils.data.DataLoader(retain_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

    best_model = finetune(
        model,
        train_loader,
        test_loader=test_loader,
        epochs=100,
        weight_decay=5e-4,
        lr=0.01,
        momentum=0.9,
        use_scheduler=True,
    )

    torch.save(best_model.state_dict(), f"../cifar10_checkpoints/retain_{seed}.pt")


@ray.remote(num_gpus=0.2, num_cpus=1)
def train_and_save_first_checkpoint(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train, test = load_cifar10()

    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.to(CONFIG.DEVICE)

    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

    best_model = finetune(
        model,
        train_loader,
        test_loader=test_loader,
        epochs=100,
        weight_decay=5e-4,
        lr=0.01,
        momentum=0.9,
        use_scheduler=True,
    )

    torch.save(best_model.state_dict(), f"../cifar10_checkpoints/initial.pt")


def main(args):
    CONFIG.DEVICE = torch.device(args.device)

    train, test = load_cifar10()

    len_forget = 1000
    len_retain = len(train) - len_forget

    torch.manual_seed(0)

    shuffled_indices = torch.randperm(len(train))

    np.save(
        "../cifar10_checkpoints/forget_set.npy",
        np.array(shuffled_indices[:len_forget]),
    )

    np.save(
        "../cifar10_checkpoints/retain_set.npy",
        np.array(shuffled_indices[len_forget:]),
    )

    forget_set = torch.utils.data.Subset(train, shuffled_indices[:len_forget])
    retain_set = torch.utils.data.Subset(train, shuffled_indices[len_forget:])

    ray.init(num_gpus=1, num_cpus=12, ignore_reinit_error=True)

    # Split into forget and retain set and train checkpoints only on retain set

    task_scheduler = BlockingScheduler(5)

    # Initial cp
    task_scheduler.schedule(train_and_save_first_checkpoint, 0)

    # CP trained on retain set
    for i in range(args.num_models):
        task_scheduler.schedule(
            train_and_save_checkpoint, shuffled_indices[len_forget:], i
        )

    ray.get(task_scheduler.scheduled_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", required=False)
    parser.add_argument("--device", type=str, default="cuda", required=False)
    parser.add_argument("--num_models", type=int, default=100, required=False)
    args = parser.parse_args()
    main(args)
