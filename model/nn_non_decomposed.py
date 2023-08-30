import torch.nn as nn
from model.common import get_resnet18_model

class MultiConceptNonDecomposed(nn.Module):
    def __init__(self, dim):
        super(MultiConceptNonDecomposed, self).__init__()
        self.model = get_resnet18_model(dim=512)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        return self.output(self.model(x))
        