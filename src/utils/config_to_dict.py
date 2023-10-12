from yacs.config import CfgNode as CN


# Source: https://github.com/rbgirshick/yacs/issues/19
def convert_cfg_to_dict(cfg_node, key_list=None):
    """ Convert a config node to dictionary """

    if key_list is None:
        key_list = []
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict
