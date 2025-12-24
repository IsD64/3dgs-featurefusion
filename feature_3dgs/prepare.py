from gaussian_splatting.prepare import basemodes, shliftmodes, colmap_init, prepare_trainer
from gaussian_splatting.trainer.extensions import ScaleRegularizeTrainerWrapper
from gaussian_splatting.dataset import CameraDataset
from feature_3dgs import FeatureGaussian
from feature_3dgs.trainer import FeatureTrainer
from feature_3dgs.extractor import AbstractFeatureExtractor

def prepare_feature_gaussians(
        sh_degree: int,
        source: str,
        device: str,
        trainable_camera: bool = False,
        load_ply: str = None
) -> FeatureGaussian:
    assert trainable_camera == False, "Camera trainable not implemented!"
    gaussians = FeatureGaussian(sh_degree).to(device)
    gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
    return gaussians

# TODO
def prepare_feature_trainer(
        gaussians: FeatureGaussian,
        extractor: AbstractFeatureExtractor,
        *args, **kwargs
) -> FeatureTrainer:
    trainer = prepare_trainer(gaussians=gaussians, *args, **kwargs)
    feature_trainer = FeatureTrainer(base_trainer=trainer, extractor=extractor)
    return feature_trainer

# TODO
def prepare_feature_extractor():
    pass