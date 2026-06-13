from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Type
import numpy as np
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig, ColmapDataParser
import os


@dataclass
class RobustNeRFConfig(ColmapDataParserConfig):
    _target: Type = field(default_factory=lambda: RobustNeRF)
    """Target class to instantiate."""
    scene: Literal["android", "crab2", "statue", "yoda",] = "android"
    """Scene name."""


@dataclass
class RobustNeRF(ColmapDataParser):
    config: RobustNeRFConfig

    def __init__(self, config: RobustNeRFConfig):
        super().__init__(config)

    def _get_image_indices(self, image_filenames, split):
        filenames = [os.path.basename(image_filename) for image_filename in image_filenames]
        groups = {
            mode: sorted([filename for filename in filenames if mode in filename])
            for mode in ['clean', 'clutter', 'extra']
        }
        rng = np.random.default_rng(0)
        if len(groups['extra']) > 19:
            groups['extra'] = np.sort(rng.choice(groups['extra'], 19, replace=False))

        if split == "train":
            indices = [i for i, filename in enumerate(filenames) if filename in groups['clutter']]
        elif split in ["val", "test"]:
            indices = [i for i, filename in enumerate(filenames) if filename in groups['extra']]
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        return indices
