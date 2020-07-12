"""
Josh Barrios 7/9/2020
Runs training, validation or prediction

arguments:
--save, path for the checkpoint with best accuracy.
--load, path to the checkpoint which will be loaded for inference or fine-tuning.
-m or --mode, 'train' 'val' or 'predict'
--data_path, path to the data root.
-b or --batch_size, batch size
--val_split, percent split for validation
-e or --epochs, # of epochs
--seed, global seed. If not specified it will be randomized (and printed on the log)
"""

# %%
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.optim import lr_scheduler

import dataset

# set up GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU enabled")
else:
    device = torch.device("cpu")
    print("GPU unavailable, running on CPU")

# %%

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save',
                        help='path for the checkpoint with best accuracy.'
                             'Checkpoint for each epoch will be saved with suffix .<number of epoch>')
    parser.add_argument('--load',
                        help='path to the checkpoint which will be loaded for inference or fine-tuning')
    parser.add_argument('-m', '--mode', default='train', choices=('train', 'val', 'predict'))
    parser.add_argument('--data_path', type=Path, default=Path('/media/userman/249272E19272B6C0/Documents and Settings/'
                                                               'jbarr/Documents/Douglass Lab/2020/2p behavior data/'
                                                               'training_data'),
                        help='path to the data root.')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int,
                        help='global seed (for weight initialization, data sampling, etc.). '
                             'If not specified it will be randomized (and printed on the log)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate decay amount')
    parser.add_argument('--lr_decay', type=float, default=0.001,
                        help='Learning rate decay amount')
    parser.add_argument('--lr_decay_step', type=int, default=5,
                        help='Number of epochs between lr decay')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Shuffle dataset before train/val split')

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

def train(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    criterion = nn.MSELoss()

    train_loader, validation_loader = dataset.get_train_val_loader(args)

    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()

        # Train set
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            output = model(images).to(torch.double)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update on training progress every 5th iteration
            if (i + 1) % 5 == 0:
                print(
                    f'epoch {epoch + 1}/{args.epochs}, step {i + 1}/{n_iterations},  loss {loss}')

        # Validation set
        for i, (images, targets) in enumerate(val_loader):
            model.eval()
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)

            loss = criterion(output, targets)

            # Update on validation loss
            print(
                f'===== VALIDATION epoch {epoch + 1}/{args.epochs}, step {i + 1}/{n_iterations},'
                'validation loss {loss} =====')

        exp_lr_scheduler.step()


# %%
def main(args):
    model = ModelAndLoss(args).cuda()
    logging.info('Model:\n{}'.format(str(model)))

    if args.load is not None:
        logging.info('Loading model from {}'.format(args.load))
        model.load_state_dict(torch.load(str(args.load)))

    if args.mode in ['train', 'val']:
        train(args, model)
    elif args.mode == 'predict':
        predict(args, model)
    else:
        assert 0
