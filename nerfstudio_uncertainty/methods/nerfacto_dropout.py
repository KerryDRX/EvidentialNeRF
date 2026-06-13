from nerfstudio_uncertainty.methods.nerfacto import *
from nerfstudio_uncertainty.metrics import evaluate_normal


@dataclass
class NerfactoDropoutConfig(NerfactoConfig):
    _target: Type = field(default_factory=lambda: NerfactoDropoutModel)

    implementation: Literal["tcnn", "torch"] = "torch"
    """Which implementation to use for the model."""
    p: float = 0.2
    """Dropout probability."""

class NerfactoDropoutModel(NerfactoModel):
    config: NerfactoDropoutConfig

    def populate_modules(self):
        super().populate_modules()

        scene_contraction = None if self.config.disable_scene_contraction else SceneContraction(order=float("inf"))
        self.field = NerfactoDropoutField(
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
            p=self.config.p,
        )
        self.proposal_networks = nn.ModuleList([
            HashMLPDensityFieldDropout(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)],
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
                p=self.config.p,
            )
            for i in range(self.config.num_proposal_iterations)
        ])
        self.density_fns = [network.density_fn for network in self.proposal_networks]

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = self.renderer_rgb.blend_background(batch["image"].to(self.device))
        predicted_rgb = outputs["rgb"]

        images_dict = {"rgb": predicted_rgb, "accumulation": outputs["accumulation"]}

        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]  # [1, C, H, W]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]  # [1, C, H, W]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)
        metrics_dict = {"psnr": psnr.item(), "ssim": ssim.item(), "lpips": lpips.item()}  # type: ignore

        uncertainty3 = outputs["U"].clamp(1e-6)  # [H, W, 3]
        uncertainty3 = uncertainty3.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        uncertainty1 = uncertainty3.mean(1, keepdim=True)  # [1, 1, H, W]
        metrics_dict.update(evaluate_normal(
            ground_truth3=gt_rgb,
            prediction3=predicted_rgb,
            uncertainty1=uncertainty1,
        ))
        images_dict.update({"U": uncertainty1.squeeze(0).squeeze(0).unsqueeze(-1)})  # [H, W, 1]

        return metrics_dict, images_dict

class NerfactoDropoutField(NerfactoField):
    def __init__(self, *args, **kwargs) -> None:
        p = kwargs.pop("p")
        super().__init__(*args, **kwargs)
        self.mlp_base = MLPWithHashEncoding(
            num_levels=self.mlp_base.num_levels,
            min_res=self.mlp_base.min_res,
            max_res=self.mlp_base.max_res,
            log2_hashmap_size=self.mlp_base.log2_hashmap_size,
            features_per_level=self.mlp_base.features_per_level,
            num_layers=self.mlp_base.num_layers,
            layer_width=self.mlp_base.layer_width,
            out_dim=self.mlp_base.out_dim,
            activation=nn.Sequential(nn.ReLU(), nn.Dropout(p)),
            out_activation=self.mlp_base.out_activation,
            implementation=kwargs.get("implementation", "torch"),
        )
        self.mlp_head = MLP(
            in_dim=self.mlp_head.in_dim,
            num_layers=self.mlp_head.num_layers,
            layer_width=self.mlp_head.layer_width,
            out_dim=self.mlp_head.out_dim,
            activation=nn.Sequential(nn.ReLU(), nn.Dropout(p)),
            out_activation=self.mlp_head.out_activation,
            implementation=kwargs.get("implementation", "torch"),
        )

class HashMLPDensityFieldDropout(HashMLPDensityField):
    def __init__(self, *args, **kwargs) -> None:
        p = kwargs.pop("p")
        super().__init__(*args, **kwargs)
        self.mlp_base = nn.Sequential(
            self.encoding,
            MLP(
                in_dim=self.encoding.get_out_dim(),
                num_layers=kwargs.get("num_layers", 2),
                layer_width=kwargs.get("hidden_dim", 64),
                out_dim=1,
                activation=nn.Sequential(nn.ReLU(), nn.Dropout(p)),
                out_activation=None,
                implementation=kwargs.get("implementation", "torch"),
            )
        )

class NeRFDropoutPipeline(VanillaPipeline):
    @profiler.time_function
    def get_average_image_metrics(self, data_loader, image_prefix: str, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False):
        self.eval()

        def _enable_dropout(module: nn.Module) -> None:
            if isinstance(module, nn.Dropout):
                module.train()
        self.model.apply(_enable_dropout)

        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            for camera, batch in data_loader:
                inner_start = time()
                height, width = camera.height, camera.width
                num_rays = height * width

                all_outputs = [self.model.get_outputs_for_camera(camera=camera) for _ in range(5)]
                outputs = {k: torch.stack([output[k] for output in all_outputs]).mean(0) for k in all_outputs[0]}
                outputs["U"] = torch.stack([output["rgb"] for output in all_outputs]).var(0, correction=0)

                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
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
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])))

        self.train()
        return metrics_dict
