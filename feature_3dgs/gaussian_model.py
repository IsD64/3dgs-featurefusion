import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel

class FeatureGaussian(GaussianModel):
    def __init__(self, sh_degree):
        super().__init__(self, sh_degree)
        self._semantic_features = torch.empty(0)

    def capture(self):
        return(
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._semantic_features,
        )

    @property
    def get_semantic_features(self):
        return self._semantic_features

    def rewrite_semantic_feature(self, x):
        self._semantic_features = x

    def create_from_pcd(
            self,
            points: torch.Tensor,
            colors: torch.Tensor,
            semantic_feature_size: int,
            speedup: bool
    ):
        super().create_from_pcd(points, colors)
        if speedup: # speed up for Segmentation
            semantic_feature_size = int(semantic_feature_size/4)
        self._semantic_features = torch.zeros(self._xyz.shape[0], semantic_feature_size, 1).float().cuda() 
        self._semantic_features = nn.Parameter(self._semantic_features.transpose(1, 2).contiguous().requires_grad_(True))
        return self

    def update_points_add(self,
            xyz: nn.Parameter,
            features_dc: nn.Parameter,
            features_rest: nn.Parameter,
            scaling: nn.Parameter,
            rotation: nn.Parameter,
            opacity: nn.Parameter,
            semantic_features: nn.Parameter
    ):
        def is_same_prefix(attr: nn.Parameter, ref: nn.Parameter):
            return (attr[:ref.shape[0]] == ref).all()
        super().update_points_add(xyz, features_dc, features_rest, scaling, rotation, opacity)
        assert is_same_prefix(semantic_features, self._semantic_features)
        self._semantic_features = semantic_features

    def update_points_replace(
            self,
            xyz_mask: torch.Tensor, xyz: nn.Parameter,
            features_dc_mask: torch.Tensor, features_dc: nn.Parameter,
            features_rest_mask: torch.Tensor, features_rest: nn.Parameter,
            scaling_mask: torch.Tensor, scaling: nn.Parameter,
            rotation_mask: torch.Tensor, rotation: nn.Parameter,
            opacity_mask: torch.Tensor, opacity: nn.Parameter,
            semantic_features_mask: torch.Tensor, semantic_features: nn.Parameter
    ):
        super().update_points_replace(xyz_mask, xyz, features_dc_mask, features_dc, features_rest_mask, features_rest, scaling_mask, scaling, rotation_mask, rotation, opacity_mask, opacity)
        def is_same_rest(attr: nn.Parameter, ref: nn.Parameter, mask: torch.Tensor):
            return (attr[~mask, ...] == ref[~mask, ...]).all()
        assert semantic_features is None or is_same_rest(semantic_features, self._semantic_features, semantic_features_mask)
        self._semantic_features = semantic_features

    def update_points_remove(
            self,
            removed_mask: torch.Tensor,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            semantic_features
    ):
        def is_same_rest(attr: nn.Parameter, ref: nn.Parameter):
            return (attr == ref[~removed_mask, ...]).all()
        super().update_points_remove(removed_mask, xyz, features_dc, features_rest, scaling, rotation, opacity)
        assert is_same_rest(semantic_features, self._semantic_features)
        self._semantic_features = semantic_features