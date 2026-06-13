from nerfstudio_uncertainty.methods.nerfacto import *
from nerfstudio_uncertainty.metrics import evaluate_evidential
import torch.nn.functional as F


@dataclass
class NerfactoEvidentialConfig(NerfactoConfig):
    _target: Type = field(default_factory=lambda: NerfactoEvidentialModel)

    u_min: float = 1e-6
    """Minimum uncertainty"""
    lambda_mse: float = 0.0
    """MSE loss coefficient"""
    lambda_edl: float = 1.0
    """EDL loss coefficient"""
    lambda_reg: float = 1.0
    """EDL regularization coefficient"""
    warmup_steps: int = 0
    """Warmup steps"""
    view_dependent: bool = False
    """Use view-dependent uncertainty"""


class NerfactoEvidentialModel(NerfactoModel):
    config: NerfactoEvidentialConfig

    def populate_modules(self):
        super().populate_modules()
        self.field = NerfactoEvidentialField(
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
            view_dependent=self.config.view_dependent,
        )
        self.lambda_mse = self.config.lambda_mse
        self.lambda_edl = self.config.lambda_edl
        self.lambda_reg = self.config.lambda_reg

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        callbacks = []
        if training_callback_attributes.trainer is not None:
            self.max_num_iterations = training_callback_attributes.trainer.config.max_num_iterations
        if self.config.use_proposal_weight_anneal:
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step
                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        def update_loss_mult(step):
            self.step = step
            if step < self.config.warmup_steps:
                self.lambda_mse, self.lambda_edl, self.lambda_reg = 1.0, 0.0, 0.0
            else:
                self.lambda_mse, self.lambda_edl, self.lambda_reg = self.config.lambda_mse, self.config.lambda_edl, self.config.lambda_reg

        callbacks.append(TrainingCallback(
            where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            update_every_num_iters=1,
            func=update_loss_mult,
        ))

        return callbacks

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
        AU = (field_outputs["AU"] * weights2).sum(-2)  # [num_rays, 1]
        EU = (field_outputs["EU"] * weights2).sum(-2)  # [num_rays, 1]
        alpha_minus_1 = (field_outputs["alpha_minus_1"] * weights).sum(-2)  # [num_rays, 1]

        outputs = {
            "rgb": rgb,  # [num_rays, 3]
            "accumulation": accumulation,  # [num_rays, 1]
            "depth": depth,  # [num_rays, 1]
            "AU": AU,  # [num_rays, 1]
            "EU": EU,  # [num_rays, 1]
            "alpha_minus_1": alpha_minus_1,  # [num_rays, 1]

            # "density": field_outputs[FieldHeadNames.DENSITY],  # [B, 48, 1]
            # "delta": ray_samples.deltas,
            # "voxel_depth": (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2,  # [num_rays, num_samples, 1]
            # "voxel_density": field_outputs[FieldHeadNames.DENSITY],  # [num_rays, num_samples, 1]
            # "voxel_weights": weights,  # [num_rays, num_samples, 1]
            # "voxel_rgb": field_outputs[FieldHeadNames.RGB],  # [num_rays, num_samples, 3]
            # "voxel_AU": field_outputs["AU"],  # [num_rays, num_samples, 1]
            # "voxel_EU": field_outputs["EU"],  # [num_rays, num_samples, 1]
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
        if self.lambda_mse > 0:
            loss_dict["rgb_loss"] = self.lambda_mse * self.rgb_loss(gt_rgb, pred_rgb)
        if self.lambda_edl > 0:
            AU = outputs["AU"]  # [num_rays, 1]
            EU = outputs["EU"]  # [num_rays, 1]
            alpha_minus_1 = outputs["alpha_minus_1"]  # [num_rays, 1]

            nu = (AU / EU).clamp(1e-6)
            alpha_minus_1 = alpha_minus_1.clamp(1e-6)  # [num_rays, 1]
            alpha = alpha_minus_1 + 1  # [num_rays, 1]
            alpha_plus_point5 = alpha + 0.5  # [num_rays, 1]
            ae = (gt_rgb - pred_rgb).abs()  # [num_rays, 3]
            se = ae ** 2  # [num_rays, 3]
            U = AU + EU  # [num_rays, 1]
            log_U = U.log()  # [num_rays, 1]
            log_2 = torch.tensor(2.0).to(pred_rgb).log()  # []

            loss_nll_1 = - alpha * (log_2 + alpha_minus_1.log() + log_U)  # [num_rays, 1]
            loss_nll_2 = alpha.lgamma() - alpha_plus_point5.lgamma()  # [num_rays, 1]
            loss_nll_3 = alpha_plus_point5 * (se + 2.0 * alpha_minus_1 * U).log()  # [num_rays, 3]
            loss_nll = loss_nll_1.mean() + loss_nll_2.mean() + loss_nll_3.mean()
            loss_reg = ((2 * nu + alpha) * ae).mean()
            loss_dict["edl_loss"] = self.lambda_edl * (loss_nll + self.lambda_reg * loss_reg)

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
        predictions = [outputs[key] for key in ["rgb", "AU", "EU", "alpha_minus_1"]]
        gt_rgb, pred_rgb, AU, EU, alpha_minus_1 = [tensor.permute(2, 0, 1).unsqueeze(0) for tensor in [gt_rgb] + predictions]
        metrics_dict = {key: getattr(self, key)(gt_rgb, pred_rgb).item() for key in ["psnr", "ssim", "lpips"]}

        if self.lambda_edl > 0:
            metrics_dict.update(evaluate_evidential(
                ground_truth3=gt_rgb,
                prediction3=pred_rgb,
                AU1=AU,
                EU1=EU,
                alpha_minus_1=alpha_minus_1,
            ))

        return metrics_dict, images_dict  # type: ignore


class NerfactoEvidentialField(NerfactoField):
    def __init__(self, *args, **kwargs) -> None:
        self.u_min = kwargs.pop("u_min")
        self.view_dependent = kwargs.pop("view_dependent")
        super().__init__(*args, **kwargs)
        if self.view_dependent:
            self.mlp_head = MLP(
                in_dim=self.mlp_head.in_dim,
                num_layers=self.mlp_head.num_layers,
                layer_width=self.mlp_head.layer_width,
                out_dim=6,
                activation=self.mlp_head.activation,
                out_activation=None,
                implementation=kwargs.get("implementation", "tcnn"),
            )
        else:
            self.mlp_base = MLPWithHashEncoding(
                num_levels=kwargs.get("num_levels", 16),
                min_res=kwargs.get("base_res", 16),
                max_res=kwargs.get("max_res", 2048),
                log2_hashmap_size=kwargs.get("log2_hashmap_size", 19),
                features_per_level=kwargs.get("features_per_level", 2),
                num_layers=kwargs.get("num_layers", 2),
                layer_width=kwargs.get("hidden_dim", 64),
                out_dim=1 + self.geo_feat_dim + 3,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=kwargs.get("implementation", "tcnn"),
            )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        assert positions.numel() > 0, "positions is empty."
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        assert positions_flat.numel() > 0, "positions_flat is empty."
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        if self.view_dependent:
            density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        else:
            density_before_activation, base_mlp_out, AU, EU, alpha_minus_1 = torch.split(h, [1, self.geo_feat_dim, 1, 1, 1], dim=-1)
        self._density_before_activation = density_before_activation
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return (density, base_mlp_out) if self.view_dependent else (density, base_mlp_out, AU, EU, alpha_minus_1)

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Dict[FieldHeadNames, Tensor]:
        directions = get_normalized_directions(ray_samples.frustums.directions).view(-1, 3)
        out = self.mlp_head(
            torch.cat([self.direction_encoding(directions), density_embedding.view(-1, self.geo_feat_dim)], dim=-1)  # type: ignore
        ).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        if self.view_dependent:
            rgb, AU, EU, alpha_minus_1 = out.split([3, 1, 1, 1], dim=-1)
            rgb = F.sigmoid(rgb)
        else:
            rgb = out
        outputs = {FieldHeadNames.RGB: rgb}
        return (outputs, AU, EU, alpha_minus_1) if self.view_dependent else outputs

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        if self.view_dependent:
            density, density_embedding = self.get_density(ray_samples)
            field_outputs, AU, EU, alpha_minus_1 = self.get_outputs(ray_samples, density_embedding=density_embedding)
        else:
            density, density_embedding, AU, EU, alpha_minus_1 = self.get_density(ray_samples)
            field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs.update({
            FieldHeadNames.DENSITY: density,  # [num_rays, num_samples, 1]
            "AU": F.softplus(AU) + self.u_min,  # [num_rays, num_samples, 1]
            "EU": F.softplus(EU) + self.u_min,  # [num_rays, num_samples, 1]
            "alpha_minus_1": F.softplus(alpha_minus_1),  # [num_rays, num_samples, 1]
        })
        return field_outputs
