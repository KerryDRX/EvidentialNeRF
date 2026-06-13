from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio_uncertainty.dataparsers.lf import LFConfig
from nerfstudio_uncertainty.dataparsers.llff import LLFFConfig
from nerfstudio_uncertainty.dataparsers.robustnerf import RobustNeRFConfig
from nerfstudio_uncertainty.dataparsers.phototourism import PhototourismConfig


lfDataparser = DataParserSpecification(config=LFConfig())
llffDataparser = DataParserSpecification(config=LLFFConfig())
robustnerfDataparser = DataParserSpecification(config=RobustNeRFConfig())
phototourismDataparser = DataParserSpecification(config=PhototourismConfig())
