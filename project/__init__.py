from parameters import *
from PatchDataset import PatchDataset
from torch.utils.data import DataLoader
from device import select_device

if __name__ == '__main__':
    paths = get_paths()
    device = select_device()

    patches = PatchDataset(paths['out_dir'], device)

    print(len(patches))
    print(patches[0])

    dataloader = DataLoader(patches, batch_size=4, shuffle=True, num_workers=4)

