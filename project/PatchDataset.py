import os
import torch
from torch.utils.data import Dataset
from skimage import io


class PatchDataset(Dataset):
    def __init__(self, root_path, device, use_wmap=True):
        """
        The PatchDataset class is meant for loading in and
        shuffling the dataset without blowing up the available
        memory. All the constructor needs is the location of
        the dataset, the device it should be stored on and whether
        it should also load in the weight maps or not.
        """

        self.root_path = root_path
        self.device = device
        self.use_wmap = use_wmap

        self.raw_path = '{}/raw/'.format(root_path)
        self.label_path = '{}/label/'.format(root_path)
        self.wmap_path = '{}/wmap/'.format(root_path)

        self.file_names = [file for file in os.listdir(self.raw_path) if file.endswith('.tif')]

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Returns the requested sample from the dataset.
        This function is also repeatedly called when you
        iterate over the PatchLoader class.
        """
        patch_name = self.file_names[idx]

        raw = torch.from_numpy(io.imread('{}/{}'.format(self.raw_path, patch_name)))
        raw = raw.to(self.device)

        label = torch.from_numpy(io.imread('{}/{}'.format(self.label_path, patch_name)))
        label = label.to(self.device, dtype=torch.int64)

        if self.use_wmap:
            wmap = torch.from_numpy(io.imread('{}/{}'.format(self.wmap_path, patch_name)))
            wmap = wmap.to(self.device)

            sample = {'patch_name': patch_name, 'raw': raw, 'label': label, 'wmap': wmap}
        else:
            sample = {'patch_name': patch_name, 'raw': raw, 'label': label}

        return sample
