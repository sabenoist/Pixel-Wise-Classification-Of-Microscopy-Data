import torch

from parameters import *
from PatchDataset import PatchDataset


def select_device():
    if torch.cuda.is_available():
        try:
            gpu_test = torch.empty(164, 164)
            gpu_test.to(torch.device('cuda'))
            del gpu_test

            return torch.device('cuda')  # GPU
        except:
            print('Warning: GPU too old or CUDA is broken. Using CPU instead.')

            return torch.device('cpu')  # CPU
    else:
        print('Warning: Incompatible GPU found. Using CPU instead.')

        return torch.device('cpu')  # CPU


if __name__ == '__main__':
    paths = get_paths()
    device = select_device()

    patches = PatchDataset(paths['out_dir'], device)

    print(len(patches))
    print(patches[0])
