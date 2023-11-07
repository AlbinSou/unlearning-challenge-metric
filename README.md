# Neurips 2023 Unlearning Challenge Metric implementation

I attempt to provide here an implementation of a metric similar to the one proposed by the organizers of the unlearning challenge for Neurips 2023 [link to kaggle challenge](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/overview). This metric takes the loss (or some other unknown quantity) distribution of a given number of models unlearned from a single checkpoint and compares it to the loss distribution of models trained from scratch on the retain set.

Warning: The scores obtained by using the metric provided in this repo cannot be directly compared to the scores obtained on the challenge leaderboard, indeed, it is explicitely mentionned that some details of the metric are kept hidden and thus cannot be reproduced. In this repo, I use the loss as the quantity of intereset for the distribution comparison, and instead of running m attacks per sample, I run only one.

# Installation

Follow these steps for installation

```
git clone https://github.com/AlbinSou/unlearning-challenge-metric.git
cd unlearning-challenge-metric
conda create -n unlearning python=3.10
conda activate unlearning
pip install -r requirements.txt
conda env config vars set PYTHONPATH=/home/.../unlearning-challenge-metric
mkdir cifar10_checkpoints
mkdir unlearning_checkpoints
```

# Usage

In order to compute the metric, the first thing to do is to train a given number of checkpoints (in parallel) from scratch on the "retain" set, using the src/train_cifar_checkpoints.py file.
```
python train_cifar_checkpoints.py --num_models 100
```

This will save the checkpoints trained from scratch on retain as well as the checkpoint trained on the full dataset under cifar10_checkpoints. Then, you can add your unlearning method to the unlearning_functions.py file and run it with


This will run the unlearning function once
```
python create_unlearning_checkpoints.py --unlearn_func name_of_your_function --debug
```

This will run the unlearning function many times (in parallel)
```
python create_unlearning_checkpoints.py --unlearn_func name_of_your_function --num_models 100
```

Once this is done, both cifar10_checkpoints and unlearning_checkpoints will be filled with the same number of checkpoints, which can be used to compute the metric evaluating the performance of your unlearning function. To do so, run the Metric.ipynb notebook.

