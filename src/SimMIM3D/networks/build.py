from .simmim import build_simmim

def build_network(config, is_pretrain=True):
    if is_pretrain:
        return build_simmim(config)
    else:
        raise NotImplementedError("Finetuning is not implemented yet.")