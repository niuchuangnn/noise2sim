from .common_denoiser import CommonDenoiser


def build_architecture(arc_cfg_ori):
    arc_cfg = arc_cfg_ori.copy()
    arc_type = arc_cfg.pop("type")
    if arc_type == "common_denoiser":
        return CommonDenoiser(**arc_cfg)
    else:
        raise TypeError
