from nerfstudio_uncertainty.methods.nerfacto import *
from nerfstudio_uncertainty.metrics import evaluate_normal
import torch.nn.functional as F


@dataclass
class NerfactoNormalConfig(NerfactoConfig):
    _target: Type = field(default_factory=lambda: NerfactoNormalModel)

    u_min: float = 1e-6
    """Minimum uncertainty"""
    warmup_steps: int = 0
    """Warmup steps"""

class NerfactoNormalModel(NerfactoModel):
    config: NerfactoNormalConfig

    def populate_modules(self):
        super().populate_modules()
        self.field = NerfactoNormalField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            spatial_distortion=None if self.config.disable_scene_contraction else SceneContraction(order=float("inf")),
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
            u_min=self.config.u_min,
        )

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])  # [num_rays, num_samples, 1]
        weights_list.append(weights)  # [num_rays, num_samples, 1]
        ray_samples_list.append(ray_samples)

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = weights.sum(-2)  # [num_rays, 1]
        rgb = field_outputs[FieldHeadNames.RGB]  # [num_rays, num_samples, 3]
        if not self.training:
            rgb = torch.nan_to_num(rgb)  # [num_rays, num_samples, 3]

        weights[:, -1, :] = weights[:, -1, :] + (1.0 - accumulation).clamp_min(0.0)  # [num_rays, num_samples, 1]
        weights = F.normalize(weights, p=1, dim=-2)  # [num_rays, num_samples, 1]
        rgb = torch.sum(weights * rgb, dim=-2)  # [num_rays, 3]
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        assert self.config.background_color == 'last_sample'

        weights2 = weights ** 2  # [num_rays, num_samples, 1]
        U = (field_outputs["U"] * weights2).sum(-2)  # [num_rays, 1]

        outputs = {
            "rgb": rgb,  # [num_rays, 3]
            "accumulation": accumulation,  # [num_rays, 1]
            "depth": depth,  # [num_rays, 1]
            "U": U,  # [num_rays, 1]
        }
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        gt_rgb = batch["image"].to(self.device)
        pred_rgb = outputs["rgb"]

        loss_dict = dict()
        if self.step < self.config.warmup_steps:
            loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        else:
            U = outputs["U"]
            loss_dict["nll_loss"] = (((gt_rgb - pred_rgb) ** 2 / U + U.log()) / 2).mean()

        if self.training:
            loss_dict.update({
                "interlevel_loss": self.config.interlevel_loss_mult * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"]),
                "distortion_loss": self.config.distortion_loss_mult * metrics_dict["distortion"],  # type: ignore
            })
        return loss_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = self.renderer_rgb.blend_background(batch["image"].to(self.device))  # [H, W, 3]
        images_dict = outputs
        images_dict["gt"] = gt_rgb
        predictions = [outputs[key] for key in ["rgb", "U"]]
        gt_rgb, pred_rgb, U = [tensor.permute(2, 0, 1).unsqueeze(0) for tensor in [gt_rgb] + predictions]
        metrics_dict = {key: getattr(self, key)(gt_rgb, pred_rgb).item() for key in ["psnr", "ssim", "lpips"]}

        if self.step >= self.config.warmup_steps:
            metrics_dict.update(evaluate_normal(
                ground_truth3=gt_rgb,  # [1, 3, H, W]
                prediction3=pred_rgb,  # [1, 3, H, W]
                uncertainty1=U,  # [1, 1, H, W]
            ))

        return metrics_dict, images_dict  # type: ignore

class NerfactoNormalField(NerfactoField):
    def __init__(self, *args, **kwargs) -> None:
        self.u_min = kwargs.pop("u_min")
        super().__init__(*args, **kwargs)
        self.mlp_head = MLP(
            in_dim=self.mlp_head.in_dim,
            num_layers=self.mlp_head.num_layers,
            layer_width=self.mlp_head.layer_width,
            out_dim=4,
            activation=self.mlp_head.activation,
            out_activation=None,
            implementation=kwargs.get("implementation", "tcnn"),
        )

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Dict[FieldHeadNames, Tensor]:
        directions = get_normalized_directions(ray_samples.frustums.directions).view(-1, 3)
        rgb, U = self.mlp_head(
            torch.cat([self.direction_encoding(directions), density_embedding.view(-1, self.geo_feat_dim)], dim=-1)  # type: ignore
        ).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions).split([3, 1], dim=-1)
        return {
            FieldHeadNames.RGB: F.sigmoid(rgb),
            "U": F.softplus(U) + self.u_min,  # type: ignore
        }
