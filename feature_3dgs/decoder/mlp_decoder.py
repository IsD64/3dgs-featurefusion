import torch
import torch.nn as nn
from feature_3dgs.decoder import AbstractDecoder

class MLPDecoder(AbstractDecoder):
    def __init__(self, feature_out_dim: int):
        super().__init__()
        self.output_dim = feature_out_dim
        self.fc4 = nn.Linear(128, 256).cuda()

    def load_checkpoint(path: str | None = None) -> None:
        if path is None:
            return
        raise NotImplementedError

    def __call__(self, x: torch.Tensor):
        input_dim, h, w = x.shape
        x = x.permute(1, 2, 0).contiguous().view(-1, input_dim)
        x = self.fc4(x)
        x = x.view(h, w, self.output_dim).permute(2, 0, 1).contiguous
        return x