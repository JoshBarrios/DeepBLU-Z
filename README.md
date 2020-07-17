# Deep Behavior Learning Utility for Zebrafish (DeepBLU-Z)
Deep learning for pose estimation of behaving larval zebrafish

Created in Python 3.6

Run "pip install -r requirements.txt" to get started

To train a new model, try "python3 main.py -m train --save 'PATH_TO_SAVE_MODEL_HERE'

To run a prediction, try python3 main.py -m predict -t 'PATH_TO_TIF_HERE'

arguments:

--save, path for the checkpoint with best accuracy.

--load, path to the checkpoint which will be loaded for inference or fine-tuning.

-m or --mode, 'train' 'val' or 'predict'

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