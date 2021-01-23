from .convnet import ConvNet
from .mlp import MLP
from .unet import UNet, UNet2
from .dncnn import DnCNN


def build_base_module(base_cfg_ori):
    base_cfg = base_cfg_ori.copy()
    base_type =base_cfg.pop("type")
    if base_type == "unet":
        return UNet(**base_cfg)
    elif base_type == "unet2":
        return UNet2(**base_cfg)
    elif base_type == "dncnn":
        return DnCNN(**base_cfg)
    else:
        raise TypeError