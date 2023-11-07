import argparse
import random
import os

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from src.utils import BlockingScheduler, evaluate, load_cifar10
from src.unlearning_functions import *
from src.resnet import resnet18
import src.config as CONFIG

""" 
Use this script to create N 
unlearn checkpoints inside
"""

@ray.remote(num_gpus=0.2, num_cpus=1)
def launch_unlearning_method(indices_forget, indices_retain, seed, unlearn_func):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train, test = load_cifar10()
    retain_set = torch.utils.data.Subset(train, indices_retain)
    forget_set = torch.utils.data.Subset(train, indices_forget)

    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model.to(CONFIG.DEVICE)

    model.load_state_dict(
        torch.load("../cifar10_checkpoints/initial.pt", map_location=CONFIG.DEVICE)
    )

    retain_loader = torch.utils.data.DataLoader(retain_set, batch_size=64, shuffle=True)
    forget_loader = torch.utils.data.DataLoader(forget_set, batch_size=64, shuffle=True)

    unlearn_func(
        model,
        retain_loader,
        forget_loader,
    )

    torch.save(model.state_dict(), f"../unlearning_checkpoints/unlearn_{seed}.pt")


def main(args):
    CONFIG.DEVICE = torch.device(args.device)

    train, test = load_cifar10()

    ray.init(num_gpus=1, num_cpus=8, ignore_reinit_error=True)

    forget_indices = np.load("../cifar10_checkpoints/forget_set.npy")
    retain_indices = np.load("../cifar10_checkpoints/retain_set.npy")
    print("Forget size:", len(forget_indices))
    print("Retain size:", len(retain_indices))

    # Split into forget and retain set and train checkpoints only on retain set

    if args.debug:
        train, test = load_cifar10()
        retain_set = torch.utils.data.Subset(train, retain_indices)
        forget_set = torch.utils.data.Subset(train, forget_indices)

        model = resnet18()
        model.fc = nn.Linear(512, 10)
        model.to(CONFIG.DEVICE)

        model.load_state_dict(
            torch.load("../cifar10_checkpoints/initial.pt", map_location=CONFIG.DEVICE)
        )

        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=64, shuffle=True
        )
        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=64, shuffle=True
        )
        eval_loader = torch.utils.data.DataLoader(
            test, batch_size=64, shuffle=True
        )

        globals()[args.unlearn_func](
            model,
            retain_loader,
            forget_loader,
        )

        print(evaluate(eval_loader, model))
    else:

        task_scheduler = BlockingScheduler(5)
        for i in tqdm.tqdm(range(args.num_models)):
            task_scheduler.schedule(
                launch_unlearning_method, forget_indices, retain_indices, i, globals()[args.unlearn_func]
            )

        ray.get(task_scheduler.scheduled_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", required=False)
    parser.add_argument("--device", type=str, default="cuda", required=False)
    parser.add_argument("--unlearn_func", type=str, default="unlearn_1", required=False)
    parser.add_argument("--num_models", type=int, default=100, required=False)
    args = parser.parse_args()
    main(args)
