from gaussian_splatting.dataset import CameraDataset
from .yolo import available_datasets as available_yolo_datasets


available_datasets = {
    **available_yolo_datasets,
}


def build_dataset(name: str, cameras: CameraDataset, *args, **kwargs):
    return available_datasets[name](cameras, *args, **kwargs)
