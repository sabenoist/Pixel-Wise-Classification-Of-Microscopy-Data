import os
import torch
from torch.utils.data import Dataset
from skimage import io


class PatchDataset(Dataset):
    def __init__(self, root_path, device):
        self.root_path = root_path
        self.device = device

        self.raw_path = '{}/raw/'.format(root_path)
        self.label_path = '{}/label/'.format(root_path)

        self.file_names = [file for file in os.listdir(self.raw_path) if file.endswith('.tif')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        patch_name = self.file_names[idx]

        raw = torch.from_numpy(io.imread('{}/{}'.format(self.raw_path, patch_name)))
        raw = raw.to(self.device)

        label = torch.from_numpy(io.imread('{}/{}'.format(self.label_path, patch_name)))
        label = label.to(self.device)

        sample = {'patch_name': patch_name, 'raw': raw, 'label': label}

        return sample
