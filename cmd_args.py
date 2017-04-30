import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--command', default='train', choices=['train'])
parser.add_argument('--data', default='cifar10', choices=['cifar10'])
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--data-augmentation', type=bool, default=False)
parser.add_argument('--label-corrupt-prob', type=float, default=0.0)

parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--learning-rate', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--eval-full-trainset', type=bool, default=True,
                    help='Whether to re-evaluate the full train set on a fixed model, or simply ' +
                    'report the running average of training statistics')

parser.add_argument('--arch', default='wide-resnet', choices=['wide-resnet', 'mlp'])

parser.add_argument('--wrn-depth', type=int, default=28)
parser.add_argument('--wrn-widen-factor', type=int, default=1)
parser.add_argument('--wrn-droprate', type=float, default=0.0)

parser.add_argument('--mlp-spec', default='512',
                    help='mlp spec: e.g. 512x128x512 indicates 3 hidden layers')

parser.add_argument('--name', default='', help='Experiment name')


def format_experiment_name(args):
  name = args.name
  if name != '':
    name += '_'

  name += args.data + '_'
  if args.label_corrupt_prob > 0:
    name += 'corrupt%g_' % args.label_corrupt_prob

  name += args.arch
  if args.arch == 'wide-resnet':
    dropmark = '' if args.wrn_droprate == 0 else ('-dr%g' % args.wrn_droprate)
    name += '{0}-{1}{2}'.format(args.wrn_depth, args.wrn_widen_factor, dropmark)
  elif args.arch == 'mlp':
    name += args.mlp_spec

  name += '_lr{0}_mmt{1}'.format(args.learning_rate, args.momentum)
  if args.weight_decay > 0:
    name += '_Wd{0}'.format(args.weight_decay)
  else:
    name += '_NoWd'
  if not args.data_augmentation:
    name += '_NoAug'

  return name


def parse_args():
  args = parser.parse_args()
  args.exp_name = format_experiment_name(args)
  return args
