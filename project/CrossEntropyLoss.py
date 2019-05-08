import torch

class CrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()


    def forward(self, outputs, labels, wmap):
        loss = 0

        for layer in range(len(outputs.shape[1])):
            output = outputs[layer]
            label = labels[layer]

            loss +=


        return loss
