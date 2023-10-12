from .simmim import build_simmim
from .unetr import build_unetr


def build_network(config, is_pretrain=True):
    if is_pretrain:
        return build_simmim(config)
    else:
        return build_unetr(config)
