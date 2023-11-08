#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn

import src.config as CONFIG
from src.resnet import resnet18
from src.train_cifar_checkpoints import evaluate, load_cifar10


def main(args):
    CONFIG.DEVICE = torch.device(args.device)

    model = resnet18()
    model.linear = nn.Linear(512, 10)
    model.to(CONFIG.DEVICE)

    model.load_state_dict(torch.load(args.checkpoint_name))

    train, test = load_cifar10()
    test_loader = torch.utils.data.DataLoader(test, batch_size=64)

    acc = evaluate(test_loader, model)

    print(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_name", type=str)
    parser.add_argument("--device", type=str, default="cuda", required=False)
    args = parser.parse_args()
    main(args)
