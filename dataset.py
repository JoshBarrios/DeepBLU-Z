'''
Josh Barrios 7/9/2020
Defining behavior video dataset with tracking points.
Defining training and validation dataloaders.

'''
# %%



# %% Define dataset
# define normalization function for prepping input to resnet18
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class ImagesDS(Dataset):

    def __init__(self, root_dir):
        super().__init__()
        self.root = Path(root_dir)

        # Get images from hdf file
        hdf_path = Path(self.root, 'images_ds.h5')
        hdf_file = h5py.File(hdf_path, 'r')
        for gname, group in hdf_file.items():
            for dname, ds in group.items():
                self.images = ds

        # Get tracking points from csv
        trck_pts = genfromtxt(Path(root_dir, 'trck_pts.csv'), delimiter=',')
        trck_pts = np.transpose(trck_pts)
        trck_pts = trck_pts.reshape(np.int(trck_pts.shape[0] / 8), 8, 2)
        self.trck_pts = trck_pts

    def __getitem__(self, index):

        # Get image from hdf5 dataset
        image = self.images[index, :, :]
        image = np.transpose(image)
        h = image.shape[0]
        w = image.shape[1]
        # Simulate RGB 3-channel image
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Image.fromarray(np.uint8(image))
        image = F.to_tensor(image)
        #image  = image.to(torch.double)
        image = normalize(image)

        # targets are the tracking point locations normalized to W and H of the image
        targets_ = self.trck_pts[index, :, :]
        targets = np.double(np.zeros(16))
        targets[0:8] = targets_[:, 0] / h
        targets[8:16] = targets_[:, 1] / w
        targets = torch.tensor(targets)

        return image, targets

    def __len__(self):
        return self.trck_pts.shape[0]