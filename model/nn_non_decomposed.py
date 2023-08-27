import torch.nn as nn
from model.common import get_resnet18_model

class MultiConceptNonDecomposed(nn.Module):
    def __init__(self, dim):
        super(MultiConceptNonDecomposed, self).__init__()
        self.model = get_resnet18_model(dim=dim)

    def forward(self, x):
        return self.model(x)
        