import torch.nn as nn
from .resnet import get_resnet18_model

class MultiConceptNonDecomposed(nn.Module):
    def __init__(self, dim, device = "cpu"):
        super(MultiConceptNonDecomposed, self).__init__()
        self.model = get_resnet18_model(dim=dim * 2)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim*2, int(dim*1.5)),
            nn.ReLU(),
            nn.Linear(int(dim*1.5), dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.model(x)
        x = self.output(x)
        return x 