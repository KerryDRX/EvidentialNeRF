from __future__ import annotations
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio_uncertainty.methods.nerfacto import NerfactoConfig, NerfactoPipeline
from nerfstudio_uncertainty.methods.nerfacto_normal import NerfactoNormalConfig
from nerfstudio_uncertainty.methods.nerfacto_mol import NerfactoMoLConfig
from nerfstudio_uncertainty.methods.nerfacto_dropout import NerfactoDropoutConfig, NeRFDropoutPipeline
from nerfstudio_uncertainty.methods.nerfacto_evidential import NerfactoEvidentialConfig


def specification_nerfacto(method_name, ModelConfig, Pipeline):
    return MethodSpecification(
        config=TrainerConfig(
            method_name=method_name,
            steps_per_eval_batch=0,
            steps_per_eval_image=0,
            mixed_precision=True,
            pipeline=VanillaPipelineConfig(
                _target=Pipeline,
                datamanager=ParallelDataManagerConfig(
                    train_num_rays_per_batch=4096,
                    eval_num_rays_per_batch=4096,
                    images_on_gpu=True,
                    masks_on_gpu=True,
                ),
                model=ModelConfig(
                    eval_num_rays_per_chunk=1 << 15,
                ),
            ),
            optimizers={
                "proposal_networks": {
                    "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
                },
                "fields": {
                    "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                    "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
                },
            },
            timestamp="main",
            vis="none",
        ),
        description=f"{method_name} method.",
    )

nerfacto_method = specification_nerfacto("nerfacto", NerfactoConfig, NerfactoPipeline)
nerfacto_dropout_method = specification_nerfacto("nerfacto-dropout", NerfactoDropoutConfig, NeRFDropoutPipeline)
nerfacto_normal_method = specification_nerfacto("nerfacto-normal", NerfactoNormalConfig, NerfactoPipeline)
nerfacto_mol_method = specification_nerfacto("nerfacto-mol", NerfactoMoLConfig, NerfactoPipeline)
nerfacto_evidential_method = specification_nerfacto("nerfacto-evidential", NerfactoEvidentialConfig, NerfactoPipeline)
