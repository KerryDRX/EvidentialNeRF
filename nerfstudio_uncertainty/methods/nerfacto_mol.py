from nerfstudio_uncertainty.methods.nerfacto import *
from nerfstudio_uncertainty.metrics import evaluate_mol
import torch.nn.functional as F


@dataclass
class NerfactoMoLConfig(NerfactoConfig):
    _target: Type = field(default_factory=lambda: NerfactoMoLModel)

    b_min: float = 1e-3
    """Minimum scale"""
    warmup_steps: int = 0
    """Warmup steps"""

class NerfactoMoLModel(NerfactoModel):
    config: NerfactoMoLConfig

    def populate_modules(self):
        super().populate_modules()
        self.field = NerfactoMoLField(
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
            b_min=self.config.b_min,
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

        outputs = {
            "rgb": rgb,  # [num_rays, 3]
            "accumulation": accumulation,  # [num_rays, 1]
            "depth": depth,  # [num_rays, 1]
            "weight": weights,  # [num_rays, num_samples, 1]
            "mu": field_outputs[FieldHeadNames.RGB],  # [num_rays, num_samples, 3]
            "b": field_outputs["b"],  # [num_rays, num_samples, 1]
        }
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        gt_rgb = batch["image"].to(self.device)  # [num_rays, 3]
        pred_rgb = outputs["rgb"]  # [num_rays, 3]

        loss_dict = dict()
        if self.step < self.config.warmup_steps:
            loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        else:
            w = outputs["weight"]  # [num_rays, num_samples, 1]
            mu = outputs["mu"]  # [num_rays, num_samples, 3]
            b = outputs["b"].expand_as(mu)  # [num_rays, num_samples, 3]
            log_pdf = -(gt_rgb.unsqueeze(-2) - mu).abs() / b - (2.0 * b).log()  # [num_rays, num_samples, 3]
            log_w = (w + 1e-12).log()  # [num_rays, num_samples, 1]
            log_likelihood = torch.logsumexp(log_w + log_pdf, dim=-2)  # [num_rays, 3]
            loss_dict["nll_loss"] = -log_likelihood.mean()

        if self.training:
            loss_dict.update({
                "interlevel_loss": self.config.interlevel_loss_mult * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"]),
                "distortion_loss": self.config.distortion_loss_mult * metrics_dict["distortion"],  # type: ignore
            })
        return loss_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = self.renderer_rgb.blend_background(batch["image"].to(self.device))  # [H, W, 3]
        images_dict = {k: v for k, v in outputs.items() if k not in ["weight", "mu", "b"]}
        images_dict["gt"] = gt_rgb
        predictions = [outputs[key] for key in ["rgb"]]
        gt_rgb, pred_rgb = [tensor.permute(2, 0, 1).unsqueeze(0) for tensor in [gt_rgb] + predictions]
        metrics_dict = {key: getattr(self, key)(gt_rgb, pred_rgb).item() for key in ["psnr", "ssim", "lpips"]}

        if self.step >= self.config.warmup_steps:
            weight = outputs["weight"]
            if weight.ndim == 4 and weight.shape[-1] == 1:
                weight = weight.squeeze(-1)
            H, W, num_samples = weight.shape
            mu = outputs["mu"].view(H, W, num_samples, 3)
            b = outputs["b"].view(H, W, num_samples, 1).expand_as(mu)

            w_metric = weight.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, K]
            mu_metric = mu.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, H, W, K]
            b_metric = b.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, H, W, K]

            metrics_dict.update(evaluate_mol(
                ground_truth3=gt_rgb,  # [1, 3, H, W]
                prediction3=pred_rgb,  # [1, 3, H, W]
                w=w_metric,  # [1, 1, H, W, K]
                mu=mu_metric,  # [1, 3, H, W, K]
                b=b_metric,  # [1, 3, H, W, K]
            ))

            uncertainty3 = (w_metric * (2.0 * b_metric ** 2 + mu_metric ** 2)).sum(-1) - pred_rgb ** 2  # [1, 3, H, W]
            uncertainty1 = uncertainty3.mean(1, keepdim=True)  # [1, 1, H, W]
            images_dict.update({"U": uncertainty1.squeeze(0).permute(1, 2, 0)})  # [H, W, 1]

        return metrics_dict, images_dict

class NerfactoMoLField(NerfactoField):
    def __init__(self, *args, **kwargs) -> None:
        self.b_min = kwargs.pop("b_min")
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
        rgb, b = self.mlp_head(
            torch.cat([self.direction_encoding(directions), density_embedding.view(-1, self.geo_feat_dim)], dim=-1)  # type: ignore
        ).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions).split([3, 1], dim=-1)
        return {
            FieldHeadNames.RGB: F.sigmoid(rgb),
            "b": F.softplus(b) + self.b_min,  # type: ignore
        }
