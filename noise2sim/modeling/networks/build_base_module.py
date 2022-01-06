from .unet import UNet2


def build_base_module(base_cfg_ori):
    base_cfg = base_cfg_ori.copy()
    base_type =base_cfg.pop("type")
    if base_type == "unet2":
        return UNet2(**base_cfg)
    else:
        raise TypeError