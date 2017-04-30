This repo contains simple demo code for the following paper, to train over-parameterized models on random label CIFAR-10 datasets.

> Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. *Understanding deep learning requires rethinking generalization*. International Conference on Learning Representations (ICLR), 2017. [[arXiv:1611.03530](https://arxiv.org/abs/1611.03530)].

The original code for the paper belongs to Google, and there seems to be no need to go through all the legal processes to open source them as the experiments are rather straightforward to implement and reproduce.

However, since people have been asking for the code. In order for people to easily get started, I created this repo to demonstrate how easy it is to take an existing implementation of commonly used successful models (here we use [an implementation of Wide Resnets in pytorch](https://github.com/xternalz/WideResNet-pytorch)) and fit them to random labels. We are not trying to reproduce exactly the same experiments (e.g. with Inception and Alexnet and corresponding hyper parameters) in the paper. This repo is completely written from scratch, and has no association with Google.

If you are trying to reproduce the *fitting random label* results yourself but have some troubles, here are some tips that might be helpful:

- Try to run the examples in this repo directly (pytorch is needed).
- Check if the randomization of labels is fixed and consistent throughout epochs. Check if your data augmentation is turned off. It is also good to start playing with all the other regularizers (e.g. dropout, weight decay) turned off.
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
000: Acc-tr: 58.49, Acc-val: 56.92, L-tr: 1.1574, L-val: 1.1857
001: Acc-tr: 69.64, Acc-val: 68.24, L-tr: 0.8488, L-val: 0.8980
002: Acc-tr: 76.29, Acc-val: 73.24, L-tr: 0.6726, L-val: 0.7650
003: Acc-tr: 73.34, Acc-val: 70.56, L-tr: 0.7899, L-val: 0.9145
004: Acc-tr: 82.07, Acc-val: 77.42, L-tr: 0.5031, L-val: 0.6565
005: Acc-tr: 84.26, Acc-val: 79.33, L-tr: 0.4427, L-val: 0.6233
006: Acc-tr: 86.45, Acc-val: 79.70, L-tr: 0.3990, L-val: 0.6115
007: Acc-tr: 85.84, Acc-val: 78.43, L-tr: 0.3901, L-val: 0.6308
008: Acc-tr: 89.13, Acc-val: 81.18, L-tr: 0.3051, L-val: 0.5856
009: Acc-tr: 89.21, Acc-val: 80.54, L-tr: 0.3173, L-val: 0.6035
010: Acc-tr: 83.52, Acc-val: 75.90, L-tr: 0.5089, L-val: 0.8521
...
144: Acc-tr: 97.50, Acc-val: 82.48, L-tr: 0.0741, L-val: 0.7665
145: Acc-tr: 94.48, Acc-val: 80.28, L-tr: 0.1638, L-val: 0.9234
146: Acc-tr: 97.09, Acc-val: 82.18, L-tr: 0.0865, L-val: 0.7689
147: Acc-tr: 90.01, Acc-val: 76.85, L-tr: 0.3291, L-val: 1.1154
148: Acc-tr: 96.03, Acc-val: 81.72, L-tr: 0.1148, L-val: 0.7589
149: Acc-tr: 94.11, Acc-val: 80.31, L-tr: 0.1755, L-val: 0.9136
150: Acc-tr: 99.97, Acc-val: 85.87, L-tr: 0.0061, L-val: 0.5876
151: Acc-tr: 100.00, Acc-val: 86.31, L-tr: 0.0034, L-val: 0.5824
152: Acc-tr: 100.00, Acc-val: 86.25, L-tr: 0.0025, L-val: 0.5874
153: Acc-tr: 100.00, Acc-val: 86.39, L-tr: 0.0019, L-val: 0.5952
154: Acc-tr: 100.00, Acc-val: 86.37, L-tr: 0.0016, L-val: 0.5955
155: Acc-tr: 100.00, Acc-val: 86.43, L-tr: 0.0015, L-val: 0.5955
156: Acc-tr: 100.00, Acc-val: 86.34, L-tr: 0.0012, L-val: 0.5986
157: Acc-tr: 100.00, Acc-val: 86.25, L-tr: 0.0011, L-val: 0.5997
158: Acc-tr: 100.00, Acc-val: 86.40, L-tr: 0.0011, L-val: 0.5988
159: Acc-tr: 100.00, Acc-val: 86.35, L-tr: 0.0010, L-val: 0.5992
160: Acc-tr: 100.00, Acc-val: 86.34, L-tr: 0.0009, L-val: 0.6083
161: Acc-tr: 100.00, Acc-val: 86.51, L-tr: 0.0008, L-val: 0.6033
162: Acc-tr: 100.00, Acc-val: 86.39, L-tr: 0.0008, L-val: 0.6042
163: Acc-tr: 100.00, Acc-val: 86.47, L-tr: 0.0008, L-val: 0.6019
164: Acc-tr: 100.00, Acc-val: 86.39, L-tr: 0.0007, L-val: 0.6028
165: Acc-tr: 100.00, Acc-val: 86.42, L-tr: 0.0007, L-val: 0.6066
166: Acc-tr: 100.00, Acc-val: 86.44, L-tr: 0.0007, L-val: 0.6017
167: Acc-tr: 100.00, Acc-val: 86.52, L-tr: 0.0006, L-val: 0.6050
...
293: Acc-tr: 100.00, Acc-val: 86.40, L-tr: 0.0005, L-val: 0.5625
294: Acc-tr: 100.00, Acc-val: 86.49, L-tr: 0.0004, L-val: 0.5624
295: Acc-tr: 100.00, Acc-val: 86.38, L-tr: 0.0004, L-val: 0.5619
296: Acc-tr: 100.00, Acc-val: 86.44, L-tr: 0.0004, L-val: 0.5651
297: Acc-tr: 100.00, Acc-val: 86.32, L-tr: 0.0005, L-val: 0.5649
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
000: Acc-tr: 10.30, Acc-val:  9.91, L-tr: 2.3105, L-val: 2.3129
001: Acc-tr: 10.55, Acc-val:  9.86, L-tr: 2.3038, L-val: 2.3051
002: Acc-tr: 11.07, Acc-val: 10.28, L-tr: 2.3023, L-val: 2.3052
003: Acc-tr: 10.96, Acc-val: 10.16, L-tr: 2.3002, L-val: 2.3043
004: Acc-tr: 10.87, Acc-val: 10.04, L-tr: 2.2993, L-val: 2.3054
005: Acc-tr: 11.40, Acc-val:  9.61, L-tr: 2.3000, L-val: 2.3071
006: Acc-tr: 11.70, Acc-val:  9.81, L-tr: 2.2974, L-val: 2.3062
007: Acc-tr: 12.45, Acc-val:  9.63, L-tr: 2.2941, L-val: 2.3081
008: Acc-tr: 12.27, Acc-val:  9.73, L-tr: 2.2929, L-val: 2.3088
009: Acc-tr: 13.29, Acc-val:  9.73, L-tr: 2.2888, L-val: 2.3128
010: Acc-tr: 14.44, Acc-val:  9.69, L-tr: 2.2826, L-val: 2.3103
...
038: Acc-tr: 48.70, Acc-val: 10.00, L-tr: 1.4724, L-val: 3.7426
039: Acc-tr: 54.72, Acc-val:  9.83, L-tr: 1.3039, L-val: 3.6905
040: Acc-tr: 63.80, Acc-val: 10.34, L-tr: 1.1148, L-val: 3.6388
041: Acc-tr: 63.55, Acc-val: 10.08, L-tr: 1.0545, L-val: 3.6621
...
147: Acc-tr: 67.70, Acc-val: 10.24, L-tr: 1.0133, L-val: 6.0229
148: Acc-tr: 76.09, Acc-val:  9.82, L-tr: 0.7517, L-val: 5.8342
149: Acc-tr: 77.35, Acc-val:  9.75, L-tr: 0.6599, L-val: 5.4791
150: Acc-tr: 99.00, Acc-val:  9.90, L-tr: 0.0809, L-val: 5.5166
151: Acc-tr: 99.87, Acc-val:  9.92, L-tr: 0.0398, L-val: 5.9411
152: Acc-tr: 99.97, Acc-val:  9.89, L-tr: 0.0240, L-val: 6.2846
153: Acc-tr: 99.98, Acc-val:  9.94, L-tr: 0.0172, L-val: 6.5499
154: Acc-tr: 100.00, Acc-val:  9.94, L-tr: 0.0121, L-val: 6.7529
155: Acc-tr: 100.00, Acc-val: 10.16, L-tr: 0.0093, L-val: 6.9280
156: Acc-tr: 100.00, Acc-val: 10.03, L-tr: 0.0080, L-val: 7.0225
157: Acc-tr: 100.00, Acc-val: 10.07, L-tr: 0.0065, L-val: 7.1910
158: Acc-tr: 100.00, Acc-val:  9.97, L-tr: 0.0056, L-val: 7.2699
159: Acc-tr: 100.00, Acc-val: 10.04, L-tr: 0.0048, L-val: 7.3477
160: Acc-tr: 100.00, Acc-val: 10.06, L-tr: 0.0042, L-val: 7.4414
161: Acc-tr: 100.00, Acc-val:  9.95, L-tr: 0.0037, L-val: 7.5141
162: Acc-tr: 100.00, Acc-val: 10.00, L-tr: 0.0034, L-val: 7.5898
163: Acc-tr: 100.00, Acc-val:  9.95, L-tr: 0.0032, L-val: 7.6107
164: Acc-tr: 100.00, Acc-val: 10.16, L-tr: 0.0030, L-val: 7.6497
165: Acc-tr: 100.00, Acc-val:  9.92, L-tr: 0.0027, L-val: 7.6929
166: Acc-tr: 100.00, Acc-val:  9.98, L-tr: 0.0026, L-val: 7.7483
167: Acc-tr: 100.00, Acc-val: 10.06, L-tr: 0.0026, L-val: 7.7285
168: Acc-tr: 100.00, Acc-val: 10.01, L-tr: 0.0023, L-val: 7.7625
169: Acc-tr: 100.00, Acc-val:  9.89, L-tr: 0.0022, L-val: 7.7802
170: Acc-tr: 100.00, Acc-val:  9.77, L-tr: 0.0021, L-val: 7.7844
171: Acc-tr: 100.00, Acc-val: 10.04, L-tr: 0.0020, L-val: 7.8608
172: Acc-tr: 100.00, Acc-val:  9.95, L-tr: 0.0019, L-val: 7.8329
173: Acc-tr: 100.00, Acc-val:  9.84, L-tr: 0.0018, L-val: 7.8759
...
294: Acc-tr: 100.00, Acc-val: 10.02, L-tr: 0.0009, L-val: 7.6708
295: Acc-tr: 100.00, Acc-val: 10.11, L-tr: 0.0009, L-val: 7.6783
296: Acc-tr: 100.00, Acc-val: 10.02, L-tr: 0.0010, L-val: 7.6486
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

