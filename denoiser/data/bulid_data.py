from denoiser.data.datasets.data_lmdb import LMDB
from denoiser.data.datasets.bsd_npy import BSDNPY


def build_dataset(data_cfg_ori):
    data_cfg = data_cfg_ori.copy()
    data_type = data_cfg.pop("type")
    dataset = None

    if data_type == "lmdb":
        dataset = LMDB(**data_cfg)
    elif data_type == "bsd_npy":
        dataset = BSDNPY(**data_cfg)
    else:
        assert TypeError

    return dataset