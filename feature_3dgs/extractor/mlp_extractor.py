import torch
from .abc import AbstractFeatureExtractor

class MLPExtractor(AbstractFeatureExtractor):
    def __init__(self):
        pass

    def extract(image: torch.Tensor) -> torch.Tensor:
        pass