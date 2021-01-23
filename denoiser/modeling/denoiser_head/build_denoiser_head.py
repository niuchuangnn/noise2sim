from .denoiser_head import DenoiserHead


def build_denoiser_head(cfg):

    return DenoiserHead(**cfg)