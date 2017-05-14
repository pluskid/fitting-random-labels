- [Code organization](#code-organization)
- [Command examples](#command-examples)
- [SGD implicit regularization](#sgd-implicit-regularization)

This repo contains simple demo code for the following paper, to train over-parameterized models on random label CIFAR-10 datasets.

> Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. *Understanding deep learning requires rethinking generalization*. International Conference on Learning Representations (ICLR), 2017. [[arXiv:1611.03530](https://arxiv.org/abs/1611.03530)].

The original code for the paper belongs to Google, and there seems to be no need to go through all the legal processes to open source them as the experiments are rather straightforward to implement and reproduce.

However, since people have been asking for the code. In order for people to easily get started, I created this repo to demonstrate how one can take an existing implementation of commonly used successful models (here we use [an implementation of Wide Resnets in pytorch](https://github.com/xternalz/WideResNet-pytorch)) and fit them to random labels. We are not trying to reproduce exactly the same experiments (e.g. with Inception and Alexnet and corresponding hyper parameters) in the paper. This repo is completely written from scratch, and has no association with Google.

If you are trying to reproduce the *fitting random label* results yourself but have some troubles, here are some tips that might be helpful:

- Try to run the examples in this repo directly (pytorch is needed) and modify based on this.
- Check if the randomization of labels is fixed and consistent throughout epochs. Check if your data augmentation is turned off. It is also good to start playing with all the regularizers (e.g. dropout, weight decay) turned off.
- Try to run more epochs. Random labels could take more epochs to fit.
- If you are trying to reproduce the results on MNIST, you might need to try harder. MNIST contains digits that are very similar even in the pixel space. So assigning different (random) labels to those visually similar digits makes it quite hard to fit. So you might need to run more epochs or tune the learning rate a bit depending on the specific architecture you use.

# Code organization

- `train.py`: main command line interface and training loops.
- `model_wideresnet.py` and `model_mlp.py`: model definition for Wide Resnets and MLPs.
- `cifar10_data.py`: a thin wrapper of torchvision's CIFAR-10 dataset to support (full or partial) random label corruption.
- `cmd_args.py`: command line argument parsing.

# Command examples

Show all the program arguments
```
python train.py --help
```

## Wide Resnet on the original CIFAR-10
```
python train.py
```
and here are some sample outputs from my local run:
```
Number of parameters: 369498
000: Acc-tr:  58.49, Acc-val: 56.92, L-tr: 1.1574, L-val: 1.1857
001: Acc-tr:  69.64, Acc-val: 68.24, L-tr: 0.8488, L-val: 0.8980
002: Acc-tr:  76.29, Acc-val: 73.24, L-tr: 0.6726, L-val: 0.7650
003: Acc-tr:  73.34, Acc-val: 70.56, L-tr: 0.7899, L-val: 0.9145
004: Acc-tr:  82.07, Acc-val: 77.42, L-tr: 0.5031, L-val: 0.6565
005: Acc-tr:  84.26, Acc-val: 79.33, L-tr: 0.4427, L-val: 0.6233
...
149: Acc-tr:  94.11, Acc-val: 80.31, L-tr: 0.1755, L-val: 0.9136
150: Acc-tr:  99.97, Acc-val: 85.87, L-tr: 0.0061, L-val: 0.5876
151: Acc-tr: 100.00, Acc-val: 86.31, L-tr: 0.0034, L-val: 0.5824
152: Acc-tr: 100.00, Acc-val: 86.25, L-tr: 0.0025, L-val: 0.5874
...
166: Acc-tr: 100.00, Acc-val: 86.44, L-tr: 0.0007, L-val: 0.6017
167: Acc-tr: 100.00, Acc-val: 86.52, L-tr: 0.0006, L-val: 0.6050
...
298: Acc-tr: 100.00, Acc-val: 86.43, L-tr: 0.0005, L-val: 0.5649
299: Acc-tr: 100.00, Acc-val: 86.41, L-tr: 0.0004, L-val: 0.5636
```

## Wide Resnet on CIFAR-10 with Random Labels
```
python train.py --label-corrupt-prob=1.0
```
and here are some sample outputs from my local run:
```
Number of parameters: 369498
000: Acc-tr:  10.30, Acc-val:  9.91, L-tr: 2.3105, L-val: 2.3129
001: Acc-tr:  10.55, Acc-val:  9.86, L-tr: 2.3038, L-val: 2.3051
002: Acc-tr:  11.07, Acc-val: 10.28, L-tr: 2.3023, L-val: 2.3052
003: Acc-tr:  10.96, Acc-val: 10.16, L-tr: 2.3002, L-val: 2.3043
004: Acc-tr:  10.87, Acc-val: 10.04, L-tr: 2.2993, L-val: 2.3054
005: Acc-tr:  11.40, Acc-val:  9.61, L-tr: 2.3000, L-val: 2.3071
...
038: Acc-tr:  48.70, Acc-val: 10.00, L-tr: 1.4724, L-val: 3.7426
039: Acc-tr:  54.72, Acc-val:  9.83, L-tr: 1.3039, L-val: 3.6905
040: Acc-tr:  63.80, Acc-val: 10.34, L-tr: 1.1148, L-val: 3.6388
041: Acc-tr:  63.55, Acc-val: 10.08, L-tr: 1.0545, L-val: 3.6621
...
147: Acc-tr:  67.70, Acc-val: 10.24, L-tr: 1.0133, L-val: 6.0229
148: Acc-tr:  76.09, Acc-val:  9.82, L-tr: 0.7517, L-val: 5.8342
149: Acc-tr:  77.35, Acc-val:  9.75, L-tr: 0.6599, L-val: 5.4791
150: Acc-tr:  99.00, Acc-val:  9.90, L-tr: 0.0809, L-val: 5.5166
151: Acc-tr:  99.87, Acc-val:  9.92, L-tr: 0.0398, L-val: 5.9411
152: Acc-tr:  99.97, Acc-val:  9.89, L-tr: 0.0240, L-val: 6.2846
153: Acc-tr:  99.98, Acc-val:  9.94, L-tr: 0.0172, L-val: 6.5499
154: Acc-tr: 100.00, Acc-val:  9.94, L-tr: 0.0121, L-val: 6.7529
155: Acc-tr: 100.00, Acc-val: 10.16, L-tr: 0.0093, L-val: 6.9280
...
172: Acc-tr: 100.00, Acc-val:  9.95, L-tr: 0.0019, L-val: 7.8329
173: Acc-tr: 100.00, Acc-val:  9.84, L-tr: 0.0018, L-val: 7.8759
...
297: Acc-tr: 100.00, Acc-val:  9.97, L-tr: 0.0009, L-val: 7.6774
298: Acc-tr: 100.00, Acc-val: 10.04, L-tr: 0.0009, L-val: 7.6958
299: Acc-tr: 100.00, Acc-val:  9.99, L-tr: 0.0008, L-val: 7.7216
```

## MLP on CIFAR-10 with Random Labels
Train a MLP with 1 hidden layer and 512 hidden units with weight decay on random label CIFAR-10 dataset:
```
python train.py --arch=mlp --mlp-spec=512 --label-corrupt-prob=1.0 --learning-rate=0.01
```
and here are some sample outputs from my local run:
```
Number of parameters: 1577984
000: Acc-tr: 12.59, Acc-val: 10.15, L-tr: 2.3094, L-val: 2.3680
001: Acc-tr: 15.92, Acc-val:  9.78, L-tr: 2.2653, L-val: 2.3575
002: Acc-tr: 17.73, Acc-val:  9.36, L-tr: 2.2402, L-val: 2.3735
003: Acc-tr: 18.64, Acc-val:  9.70, L-tr: 2.2231, L-val: 2.4024
004: Acc-tr: 20.90, Acc-val:  9.52, L-tr: 2.2103, L-val: 2.4279
...
220: Acc-tr: 99.96, Acc-val:  9.54, L-tr: 0.0147, L-val: 10.2900
221: Acc-tr: 99.96, Acc-val:  9.58, L-tr: 0.0146, L-val: 10.2958
222: Acc-tr: 99.96, Acc-val:  9.54, L-tr: 0.0145, L-val: 10.2980
223: Acc-tr: 99.96, Acc-val:  9.55, L-tr: 0.0143, L-val: 10.3036
224: Acc-tr: 99.96, Acc-val:  9.57, L-tr: 0.0142, L-val: 10.3074
...
295: Acc-tr: 99.97, Acc-val:  9.55, L-tr: 0.0134, L-val: 10.3331
296: Acc-tr: 99.97, Acc-val:  9.56, L-tr: 0.0134, L-val: 10.3336
297: Acc-tr: 99.97, Acc-val:  9.57, L-tr: 0.0134, L-val: 10.3340
298: Acc-tr: 99.97, Acc-val:  9.59, L-tr: 0.0134, L-val: 10.3344
299: Acc-tr: 99.97, Acc-val:  9.57, L-tr: 0.0134, L-val: 10.3347
```

# SGD implicit regularization

In Section 5 of the paper, we talked about SGD implicit regularizing by finding the minimum norm solution in an over-parameterized linear problem with the simple square loss. A frequently asked question is how the experiments on MNIST is conducted, since it has 60,000 training examples, but only 28x28 = 784 features. As mentioned in the paper, the experiments are actually carried out by applying the "kernel trick" to Equation (3) in the paper (for both MNIST and CIFAR-10, with or without pre-processing). We are attaching a sample code for solving the MNIST raw pixel problem for reference:

```python
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

import numpy as np
import scipy.spatial
import scipy.linalg

def onehot_encode(y, n_cls=10):
    y = np.array(y, dtype=int)
    return np.array(np.eye(n_cls)[y], dtype=float)

data = np.array(mnist.data, dtype=float) / 255
labels = onehot_encode(mnist.target)

n_tr = 60000
n_tot = 70000

x_tr = data[:n_tr]; y_tr = labels[:n_tr]
x_tt = data[n_tr:n_tot]; y_tt = labels[n_tr:n_tot]

bw=2.0e-2
pdist_tr = scipy.spatial.distance.pdist(x_tr, 'sqeuclidean')
pdist_tr = scipy.spatial.distance.squareform(pdist_tr)
cdist_tt = scipy.spatial.distance.cdist(x_tt, x_tr, 'sqeuclidean')

coeff = scipy.linalg.solve(np.exp(-bw*pdist_tr), y_tr)
preds = np.argmax(np.dot(np.exp(-bw*cdist_tt), coeff), axis=1)

acc = float(np.sum(np.equal(preds, mnist.target[n_tr:n_tot]))) / (n_tot-n_tr)
print('err = %.2f%%' % (100*(1-acc)))

# err = 1.22%
```
