import torch

def select_device(force_cpu=False):
    """
    returns the device on which the UNet and the training
    should be computed. If CUDA is available, it works, and
    the program is not forced to use the CPU, the function
    will return the GPU through CUDA as the location where
    everything should be stored and computed.
    """

    if force_cpu:
        return torch.device('cpu')

    if torch.cuda.is_available():
        try:
            gpu_test = torch.empty(348, 348)
            gpu_test.to(torch.device('cuda'))
            del gpu_test

            return torch.device('cuda')  # GPU
        except:
            print('Warning: CUDA is unavailable. Using CPU instead.')

            return torch.device('cpu')  # CPU
    else:
        print('Warning: Incompatible GPU found. Using CPU instead.')

        return torch.device('cpu')  # CPU
