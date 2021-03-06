"""
Josh Barrios 7/9/2020
Deep Behavior Learning Utility for Zebrafish (DeepBLU-Z)
Runs training, validation or prediction

arguments:
--save, path for the checkpoint with best accuracy.
--load, path to the checkpoint which will be loaded for inference or fine-tuning.
-m or --mode, 'train' or 'predict'
--retrain, boolean, select to load model for further training
--data_path, path to the data root.
-t or --target, path to target image for prediction
--backbone, type of model to use for new model. Accepts all resnets, efficientnets and densenets.
-b or --batch_size, batch size
--num_pts, number of tracking points in the training data.
-b or --batch_size, batch size for training epochs
--val_split, percent split for validation
-e or --epochs, # of epochs
--seed, global seed. If not specified it will be randomized (and printed on the log)
-lr, initial learning rate
--lr_decay, amount to decay learning rate per scheduler advance
--lr_decay_step, # of epochs between scheduler advances
--shuffle, bool, shuffle dataset before split
"""

# %%
from argparse import ArgumentParser
from pathlib import Path
import logging
import sys
import time
import numpy as np
import random
import copy
from PIL import Image, ImageSequence
import imageio
# import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms

import dataset
from model import Model

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
    parser.add_argument('-l', '--load', type=str, default='./models/082020/resnet18',
                        help='path to the checkpoint which will be loaded for inference or fine-tuning')
    parser.add_argument('-m', '--mode', default='train', choices=('train', 'predict'))
    parser.add_argument('--retrain', type=bool, default=False,
                        help='load old model to continue training')
    parser.add_argument('--datapath', type=Path, default=Path('./data/training_data'),
                        help='path to the data root folder for training.')
    parser.add_argument('-t', '--target',
                        default='./data/test_data/image8.tif',
                        help='Path to target tif for prediction.')
    parser.add_argument('--backbone', default='resnet18',
                        help='backbone for the architecture.'
                             'Supported backbones: ResNets, DenseNets (from torchvision), EfficientNets. '
                             'For DenseNets, add prefix "mem-" for memory efficient version')
    parser.add_argument('--num_pts', type=int, default=8,
                        help='Number of tracking points in training data.')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int,
                        help='global seed (for weight initialization, data sampling, etc.). '
                             'If not specified it will be randomized (and printed on the log)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate decay amount')
    parser.add_argument('--lr_decay', type=float, default=0.75,
                        help='Learning rate decay amount (multiplicative)')
    parser.add_argument('--lr_decay_step', type=int, default=5,
                        help='Number of epochs between lr decay')
    parser.add_argument('--transform', type=bool, default=False,
                        help='rotate and resize input of each batch for training')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Shuffle dataset before train/val split')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Use pretrained backbone')

    args = parser.parse_args()

    if args.mode == 'train':
        assert args.save is not None
    if args.mode == 'predict':
        assert args.save is None
        assert args.target is not None

    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    return args


# %%
def setup_logging(args):
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    if args.mode == 'train':
        handlers.append(logging.FileHandler(args.save + '.log', mode='w'))
    if args.mode == 'predict':
        handlers.append(logging.FileHandler(args.load + '.output.log', mode='w'))
    logging.basicConfig(level=logging.INFO, format=head, style='{', handlers=handlers)
    logging.info('Start with arguments {}'.format(args))


# %%
def setup_determinism(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


# %% set up transformation functions

# define transform as class to allow identical transform on multiple images
class RotateAndScale:
    def __init__(self, angle, new_height, new_width):
        self.angle = angle
        self.new_height = new_height
        self.new_width = new_width

    def __call__(self, x):
        angle = self.angle
        x = transforms.functional.rotate(x, angle, expand=True)
        y = transforms.functional.resize(x, [self.new_height, self.new_width])
        return y


def transform_input(im, pts, angle, new_height, new_width):
    h = im.shape[1]
    w = im.shape[2]
    num_pts = np.int(pts.shape[0] / 2)

    warp_input = RotateAndScale(angle, new_height, new_width)

    transform_im = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(warp_input),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # for the point image, we don't need to convert to PIL
    transform_ptim = transforms.Compose([
        transforms.Lambda(warp_input),
    ])

    # Get transformed point locations by creating an image of each point and performing the same transformation on it
    new_pts = np.zeros([16], dtype='f8')
    for k in range(num_pts):
        pt_im = np.zeros([h, w], dtype='bool')
        pt_loc = [np.int(np.round(pts[k] * h)), np.int(np.round(pts[k + 8] * w))]
        # Single points often disappear when transformed, so we make a 4x4 point
        pt_im[np.int(pt_loc[0]) - 6:np.int(pt_loc[0]) + 6, np.int(pt_loc[1]) - 6:np.int(pt_loc[1]) + 6] = 1
        pt_im = Image.fromarray(pt_im)
        pt_im = transform_ptim(pt_im)
        pt_im = np.array(pt_im)
        pt = np.where(pt_im == 1)
        pt_inds = [np.int(np.round(np.mean(pt[0]))), np.int(np.round(np.mean(pt[1])))]
        new_pts[k] = pt_inds[0] / new_height
        new_pts[k + 8] = pt_inds[1] / new_width
    new_pts = torch.DoubleTensor(new_pts).unsqueeze(dim=0)
    new_im = transform_im(im).unsqueeze(dim=0)
    return new_im, new_pts


# %%
def train(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    criterion = nn.MSELoss()

    if args.retrain:
        optimizer.load_state_dict(torch.load(args.load + '_optimizer'))
        # reset lr to initial learning rate
        for g in optimizer.param_groups:
            g['lr'] = args.lr

    train_loader, validation_loader = dataset.get_train_val_loader(args)
    train_iterations = len(train_loader)
    val_iterations = len(validation_loader)

    best_loss = 10000

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Train: epoch {epoch}   learning rate: {current_lr}')

        model.train()
        optimizer.zero_grad()

        # Train set
        for i, (images, targets) in enumerate(train_loader):
            # rotate and resize batch if requested

            if args.transform:
                # Choose random rotation angle and scaling for this batch
                angle = random.choice(range(360))
                scale = random.choice(np.linspace(0.2, 2, 49))

                [new_height, new_width] = [np.int(np.round(images.size()[2] * scale)),
                                           np.int(np.round(images.size()[3] * scale))]
                # Get transformed images and targets
                for image_ind in range(len(images)):
                    if image_ind == 0:
                        new_ims, new_targets = transform_input(images[0],
                                                               targets[0],
                                                               angle,
                                                               new_height,
                                                               new_width)
                    else:
                        new_im, new_target = transform_input(images[image_ind],
                                                             targets[image_ind],
                                                             angle,
                                                             new_height,
                                                             new_width)
                        new_ims = torch.cat((new_ims, new_im), dim=0)
                        new_targets = torch.cat((new_targets, new_target), dim=0)
                images = copy.deepcopy(new_ims)
                targets = copy.deepcopy(new_targets)
                del (new_ims, new_targets)

            # For debugging - check transformed images and targets
            # testim = images[0, 0, :, :].detach().numpy()
            # testtargs = targets[0, :].detach().numpy()
            #
            # trck_pts = np.zeros([2, 8])
            # trck_pts[0, :] = testtargs[0:8] * testim.shape[0]
            # trck_pts[1, :] = testtargs[8:16] * testim.shape[1]
            # trck_pts = np.transpose(trck_pts)
            # for ind in range(trck_pts.shape[0]):
            #     pt = trck_pts[ind, :]
            #     testim[np.int(pt[0]) - 4:np.int(pt[0]) + 4, np.int(pt[1]) - 4:np.int(pt[1]) + 4] = 1
            #
            # plt.imshow(testim)
            # plt.savefig(args.save + 'test.png', dpi=300, quality=100, format='png')

            images = images.to(device)
            targets = targets.to(device)

            output = model(images).to(torch.double)

            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update on training progress every 5th iteration
            if (i + 1) % 5 == 0:
                logging.info(f'epoch {epoch + 1}/{args.epochs}, step {i + 1}/{train_iterations},  loss {loss}')

        # Validation set

        loss_log = np.zeros(len(validation_loader))

        for i, (images, targets) in enumerate(validation_loader):
            # rotate and resize if requested
            if args.transform:
                # Choose random rotation angle and scaling for this batch
                angle = random.choice(range(360))
                scale = random.choice(np.linspace(0.2, 2, 49))

                [new_height, new_width] = [np.int(np.round(images.size()[2] * scale)),
                                           np.int(np.round(images.size()[3] * scale))]
                # Get transformed images and targets
                for image_ind in range(len(images)):
                    if image_ind == 0:
                        new_ims, new_targets = transform_input(images[0],
                                                               targets[0],
                                                               angle,
                                                               new_height,
                                                               new_width)
                    else:
                        new_im, new_target = transform_input(images[image_ind],
                                                             targets[image_ind],
                                                             angle, new_height,
                                                             new_width)
                        new_ims = torch.cat((new_ims, new_im), dim=0)
                        new_targets = torch.cat((new_targets, new_target), dim=0)

                images = copy.deepcopy(new_ims)
                targets = copy.deepcopy(new_targets)
                del (new_ims, new_targets)

            model.eval()
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)

            loss = criterion(output, targets)

            loss_log[i] = loss

            # Update on validation loss
            logging.info(f'===== VALIDATION epoch {epoch + 1}/{args.epochs}, step {i + 1}/{val_iterations},'
                         f'validation loss {loss} =====')

        if np.mean(loss_log) < best_loss:
            best_loss = np.mean(loss_log)
            logging.info(f'Saving best to {args.save} with loss {best_loss}')
            torch.save(model.state_dict(), str(args.save + '/' + args.backbone))
            torch.save(optimizer.state_dict(), str(args.save + '/' + args.backbone + '_optimizer'))

        exp_lr_scheduler.step()


# %%
def predict(args, model):
    im = Image.open(args.target)

    h = np.array(im).shape[0]
    w = np.array(im).shape[1]

    # Initialize marked image array
    image_marked = np.zeros([im.n_frames, h, w], 'uint8')

    model.eval()
    # Loop over all frames, run prediction, mark frames, write result into image_marked
    index = 0
    t = time.time()
    for frame in ImageSequence.Iterator(im):
        frame_marked = np.array(frame)
        # if the model expects 3 channel image, we need to convert to "RGB"
        if model.features.conv1.in_channels == 3:
            frame = frame.convert('RGB')
            normalize = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
            frame = normalize(frame)
        else:
            normalize = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.456],
                                                                 std=[0.224])])
            frame = normalize(frame)

        frame = frame.to(device)
        frame = torch.unsqueeze(frame, 0)

        output = model(frame)
        print(output)
        # Convert length 16 tensor into 8x2 numpy tracking points
        output = output.cpu()
        output = output.detach().numpy()
        output = np.squeeze(output)

        trck_pts = np.zeros([2, 8])
        trck_pts[0, :] = output[0:8] * h
        trck_pts[1, :] = output[8:16] * w
        trck_pts[trck_pts < 0] = 0
        trck_pts = np.round(trck_pts)
        trck_pts = np.transpose(trck_pts)

        # Mark tracking points in image
        for pt in trck_pts:
            pt = np.uint16(pt)
            frame_marked[pt[0] - 4:pt[0] + 4, pt[1] - 4:pt[1] + 4] = 255
        image_marked[index, :, :] = frame_marked
        index = index + 1

    print(f'==== Predictions finished in {time.time() - t}s ====')
    im_path = args.target.split('.tif')[-2]
    im_path = im_path + '_tracked.tif'
    imageio.mimwrite(im_path, image_marked)


# %%

def main(args):
    if args.mode == 'train':
        model = Model(args).cuda()
        if args.retrain:
            model.load_state_dict(torch.load(args.load))
            logging.info(f'Loading model for training from {args.load}')
        else:
            logging.info('Building new model for training')
        logging.info(f'Model:\n{str(model)}')
        train(args, model)

        torch.save(model, Path(args.save, 'model'))

    elif args.mode == 'predict':
        # some models are saved in their entirety as a tar, others have only the state dict saved
        if '.tar' not in args.load:
            model = Model(args)
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(args.load))
            else:
                model.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
        else:
            if torch.cuda.is_available():
                model = torch.load(args.load)
            else:
                model = torch.load(args.load, map_location=torch.device('cpu'))

        model = model.to(device)
        logging.info(f'Loading model from {args.load}')
        logging.info(f'Model:\n{str(model)}')

        predict(args, model)

    else:
        assert 0


# %%
if __name__ == '__main__':
    args = parse_args()
    setup_logging(args)
    setup_determinism(args)
    main(args)
