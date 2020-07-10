'''
Josh Barrios 7/9/2020
Runs training, validation or prediction

'''

# %%
from argparse import ArgumentParser


# %%

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save',
                        help='path for the checkpoint with best accuracy. '
                             'Checkpoint for each epoch will be saved with suffix .<number of epoch>')
    parser.add_argument('--load',
                        help='path to the checkpoint which will be loaded for inference or fine-tuning')
    parser.add_argument('-m', '--mode', default='train', choices=('train', 'val', 'predict'))

    args = parser.parse_args()

    if args.mode == 'train':
        assert args.save is not None
    if args.mode == 'val':
        assert args.save is None
    if args.mode == 'predict':
        assert args.load is not None
        assert args.save is None

    return args

# %%
args = parse_args()
if args.mode == 'train':
    print('Mode is train')
if args.mode == 'val':
    print('Mode is validate')
if args.mode == 'predict':
    print('Mode is predict')