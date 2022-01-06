from .datasets.data_lmdb import LMDB
from .datasets.bsd_npy import BSDNPY
from .datasets.data_ldct import LDCT
from .datasets.data_pcct import PCCT
from .datasets.ldct_npy import LDCTNPY


def build_dataset(data_cfg_ori):
    data_cfg = data_cfg_ori.copy()
    data_type = data_cfg.pop("type")
    dataset = None

    if data_type == "lmdb":
        dataset = LMDB(**data_cfg)
    elif data_type == "bsd_npy":
        dataset = BSDNPY(**data_cfg)
    elif data_type == "ldct":
        dataset = LDCT(**data_cfg)
    elif data_type == "pcct":
        dataset = PCCT(**data_cfg)
    elif data_type =="ldctnpy":
        dataset = LDCTNPY(**data_cfg)
    else:
        assert TypeError

    return dataset