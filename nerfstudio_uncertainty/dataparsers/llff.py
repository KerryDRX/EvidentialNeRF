from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Type
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig, ColmapDataParser


@dataclass
class LLFFConfig(ColmapDataParserConfig):
    _target: Type = field(default_factory=lambda: LLFF)
    """Target class to instantiate."""
    scene: Literal["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex",] = "fern"
    """Scene name."""


@dataclass
class LLFF(ColmapDataParser):
    config: LLFFConfig

    def __init__(self, config: LLFFConfig):
        super().__init__(config)

    def _get_image_indices(self, image_filenames, split):
        if self.config.scene == "fern":
            i_train = [1, 10, 19]
            i_eval = [0, 8, 16]
        if self.config.scene == "flower":
            i_train = [1, 17, 33]
            i_eval = [0, 8, 16, 24, 32]
        if self.config.scene == "fortress":
            i_train = [1, 21, 41]
            i_eval = [0, 8, 16, 24, 32, 40]
        if self.config.scene == "horns":
            i_train = [1, 30, 61]
            i_eval = [0, 8, 16, 24, 32, 40, 48, 56]
        if self.config.scene == "leaves":
            i_train = [1, 12, 25]
            i_eval = [0, 8, 16, 24]
        if self.config.scene == "orchids":
            i_train = [1, 12, 23]
            i_eval = [0, 8, 16, 24]
        if self.config.scene == "room":
            i_train = [1, 20, 39]
            i_eval = [0, 8, 16, 24, 32, 40]
        if self.config.scene == "trex":
            i_train = [1, 28, 54]
            i_eval = [0, 8, 16, 24, 32, 40, 48]

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        return indices
