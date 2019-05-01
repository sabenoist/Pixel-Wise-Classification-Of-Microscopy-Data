import torch

def select_device(force_cpu=True):
    if force_cpu:
        return torch.device('cpu')

    if torch.cuda.is_available():
        try:
            gpu_test = torch.empty(164, 164)
            gpu_test.to(torch.device('cuda'))
            del gpu_test

            return torch.device('cuda')  # GPU
        except:
            print('Warning: CUDA is broken. Using CPU instead.')

            return torch.device('cpu')  # CPU
    else:
        print('Warning: Incompatible GPU found. Using CPU instead.')

        return torch.device('cpu')  # CPU
