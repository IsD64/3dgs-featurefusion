from gaussian_splatting import Camera
from gaussian_splatting.utils import l1_loss
from gaussian_splatting.trainer import TrainerWrapper, AbstractTrainer
class FeatureTrainer(TrainerWrapper):
    def __init__(self,base_trainer:AbstractTrainer, decoder:nn.Module):
        super().__init__(base_trainer=base_trainer)
        self.optimizer.add_param_group([{"lr":0.0001, "params":decoder.parameters()}])
        self.decoder = decoder


    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        loss = self.base_trainer.loss(out, camera)
        feature_map_3dgs = out['feature_map']
        feature_map_decoder = self.decoder(camera.ground_truth_image)
        feature_map_loss = l1_loss(feature_map_3dgs, feature_map_decoder) 
        return loss + feature_map_loss