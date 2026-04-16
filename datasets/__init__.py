"""Dataset registry: builds the dataset."""

from .spair import build as build_spair
from .spair_u import build as build_spair_u
from .pfpascal import build as build_pfpascal
from .ap10k import build as build_ap10k
from .mp100 import build as build_mp100

_BUILDERS = {
    'spair':     build_spair,
    'spair-u':   build_spair_u,
    'pf-pascal': build_pfpascal,
    'ap-10k':    build_ap10k,
    'mp-100':    build_mp100,
}


def build_dataset(dataset: str, image_set: str, args):
    if dataset not in _BUILDERS:
        raise ValueError(f'Unknown dataset: {dataset}. '
                         f'Supported: {list(_BUILDERS.keys())}')
    return _BUILDERS[dataset](image_set, args)
