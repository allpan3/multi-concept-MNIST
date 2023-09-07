import torch.nn as nn
from model.common import get_resnet18_model

class MultiConceptNonDecomposed(nn.Module):
    def __init__(self, dim, device = "cpu"):
        super(MultiConceptNonDecomposed, self).__init__()
        self.model = get_resnet18_model(dim=2048)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, dim),
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.output(self.model(x))
        