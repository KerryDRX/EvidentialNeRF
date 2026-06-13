from __future__ import annotations
from typing import Dict, Literal, Optional, Tuple, List, Type, Any, Mapping
from pathlib import Path
from time import time
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from dataclasses import dataclass, field
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler


@dataclass
class NerfactoConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: NerfactoModel)

    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """


class NerfactoModel(Model):
    config: NerfactoConfig

    def populate_modules(self):
        super().populate_modules()

        scene_contraction = None if self.config.disable_scene_contraction else SceneContraction(order=float("inf"))
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            spatial_distortion=scene_contraction,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )
        self.proposal_networks = nn.ModuleList([
            HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)],
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            for i in range(self.config.num_proposal_iterations)
        ])
        self.density_fns = [network.density_fn for network in self.proposal_networks]

        def update_schedule(step):
            return np.clip(np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]), 1, self.config.proposal_update_every)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=UniformSampler(single_jitter=self.config.use_single_jitter) if self.config.proposal_initial_sampler == "uniform" else None,
        )

        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        self.rgb_loss = MSELoss()
        self.step = 0

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "proposal_networks": list(self.proposal_networks.parameters()),
            "fields": list(self.field.parameters())
        }

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
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)  # [num_rays, num_samples, 3], [num_rays, num_samples, 1] -> [num_rays, 3]
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)  # [num_rays, 1]
        outputs = {
            "rgb": rgb,  # [num_rays, 3]
            "accumulation": accumulation,  # [num_rays, 1]
            "depth": depth,  # [num_rays, 1]
            # "weight": weights,  # [num_rays, num_samples, 1]
        }
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=batch["image"].to(self.device),
        )
        loss_dict = {"rgb_loss": self.rgb_loss(gt_rgb, pred_rgb)}
        if self.training:
            loss_dict.update({
                "interlevel_loss": self.config.interlevel_loss_mult * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"]),
                "distortion_loss": self.config.distortion_loss_mult * metrics_dict["distortion"],  # type: ignore
            })
        return loss_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        # rgb [H, W, 3]
        # accumulation [H, W, 1]
        # depth [H, W, 1]
        # weight [H, W, N]
        # prop_depth_0 [H, W, 1]
        # prop_depth_1 [H, W, 1]

        predicted_rgb = outputs["rgb"]
        gt_rgb = self.renderer_rgb.blend_background(batch["image"].to(self.device))

        images_dict = {"rgb": predicted_rgb, "accumulation": outputs["accumulation"]}

        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]  # [1, C, H, W]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]  # [1, C, H, W]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        metrics_dict = {"psnr": psnr.item(), "ssim": ssim.item(), "lpips": lpips.item()}  # type: ignore
        return metrics_dict, images_dict


class NerfactoField(Field):
    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 0.01,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion
        self.base_res = base_res
        self.average_init_density = average_init_density
        self.step = 0

        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
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
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Dict[FieldHeadNames, Tensor]:
        directions = get_normalized_directions(ray_samples.frustums.directions).view(-1, 3)
        rgb = self.mlp_head(
            torch.cat([self.direction_encoding(directions), density_embedding.view(-1, self.geo_feat_dim)], dim=-1)  # type: ignore
        ).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        outputs = {FieldHeadNames.RGB: rgb}
        return outputs

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        density, density_embedding = self.get_density(ray_samples)
        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density
        return field_outputs


class NerfactoPipeline(VanillaPipeline):
    @profiler.time_function
    def get_average_image_metrics(self, data_loader, image_prefix: str, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False):
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), MofNCompleteColumn(), transient=True) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            for camera, batch in data_loader:
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C]
                        torch.save(image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.pth")
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1
        metrics_dict = {}
        if len(metrics_dict_list) == 0:
            self.train()
            return metrics_dict
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])))
        self.train()
        return metrics_dict
