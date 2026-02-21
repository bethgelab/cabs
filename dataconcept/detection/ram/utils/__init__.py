from .metrics import get_mAP, get_PR
from .openset_utils import build_openset_label_embedding, build_openset_llm_label_embedding
import torch.nn as nn

class MLPProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPProjection, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)