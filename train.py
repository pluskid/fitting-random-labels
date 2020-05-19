from __future__ import print_function

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim

from cifar10_data import CIFAR10RandomLabels

import cmd_args
import model_mlp, model_wideresnet


def get_data_loaders(args, shuffle_train=True):
  if args.data == 'cifar10':
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.data_augmentation:
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
          ])
    else:
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader
  else:
    raise Exception('Unsupported dataset: {0}'.format(args.data))


def get_model(args):
  # create model
  if args.arch == 'wide-resnet':
    model = model_wideresnet.WideResNet(args.wrn_depth, args.num_classes,
                                        args.wrn_widen_factor,
                                        drop_rate=args.wrn_droprate)
  elif args.arch == 'mlp':
    n_units = [int(x) for x in args.mlp_spec.split('x')] # hidden dims
    n_units.append(args.num_classes)  # output dim
    n_units.insert(0, 32*32*3)        # input dim
    model = model_mlp.MLP(n_units)

  # for training on multiple GPUs.
  # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
  # model = torch.nn.DataParallel(model).cuda()
  model = model.cuda()

  return model


def train_model(args, model, train_loader, val_loader,
                start_epoch=None, epochs=None):
  cudnn.benchmark = True

  # define loss function (criterion) and pptimizer
  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

  start_epoch = start_epoch or 0
  epochs = epochs or args.epochs

  for epoch in range(start_epoch, epochs):
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    tr_loss, tr_prec1 = train_epoch(train_loader, model, criterion, optimizer, epoch, args)

    # evaluate on validation set
    val_loss, val_prec1 = validate_epoch(val_loader, model, criterion, epoch, args)

    if args.eval_full_trainset:
      tr_loss, tr_prec1 = validate_epoch(train_loader, model, criterion, epoch, args)

    logging.info('%03d: Acc-tr: %6.2f, Acc-val: %6.2f, L-tr: %6.4f, L-val: %6.4f',
                 epoch, tr_prec1, val_prec1, tr_loss, val_loss)


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
  """Train for one epoch on the training set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to train mode
  model.train()

  for i, (input, target) in enumerate(train_loader):
    target = target.cuda(non_blocking=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return losses.avg, top1.avg


def validate_epoch(val_loader, model, criterion, epoch, args):
  """Perform validation on the validation set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    target = target.cuda(non_blocking=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))

  return losses.avg, top1.avg


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
  """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
  lr = args.learning_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res


def setup_logging(args):
  import datetime
  exp_dir = os.path.join('runs', args.exp_name)
  if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)
  log_fn = os.path.join(exp_dir, "LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
  logging.basicConfig(filename=log_fn, filemode='w', level=logging.DEBUG)
  # also log into console
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)
  print('Logging into %s...' % exp_dir)


def main():
  args = cmd_args.parse_args()
  setup_logging(args)

  if args.command == 'train':
    train_loader, val_loader = get_data_loaders(args, shuffle_train=True)
    model = get_model(args)
    logging.info('Number of parameters: %d', sum([p.data.nelement() for p in model.parameters()]))
    train_model(args, model, train_loader, val_loader)


if __name__ == '__main__':
  main()

