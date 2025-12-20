import torch
from gaussian_splatting import Camera
from gaussian_splatting.trainer import TrainerWrapper
from gaussian_splatting.utils import l1_loss

# TODO add cnn_decoder and cnn_decoder_optimizer from gaussian_splatting.train.py to trainer step
class FeatureTrainer(TrainerWrapper):
    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        loss = self.base_trainer.loss(out, camera)
        feature = out["feature_map"]
        feature_loss = l1_loss(feature, camera.custom_data["feature"]) + loss
        return feature_loss
