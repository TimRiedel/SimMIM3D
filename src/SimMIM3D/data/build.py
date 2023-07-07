from .adni_pretrain_data import AdniPretrainData
from .brats_pretrain_data import BratsPretrainData
from .brats_finetune_data import BratsFinetuneData

def build_data(config, dataset, is_pretrain=True):
    if is_pretrain:
        if dataset == "brats":
            return BratsPretrainData(
                data_dir=config.DATA.DATA_DIR,
                img_size=config.DATA.IMG_SIZE,
                batch_size=config.DATA.BATCH_SIZE, 
                num_workers=config.DATA.NUM_WORKERS,
                patch_size=config.MODEL.PATCH_SIZE,
                mask_ratio=config.DATA.MASK_RATIO,
            ) 
        elif dataset == "adni":
            return AdniPretrainData(
                data_dir=config.DATA.DATA_DIR,
                img_size=config.DATA.IMG_SIZE,
                batch_size=config.DATA.BATCH_SIZE, 
                num_workers=config.DATA.NUM_WORKERS,
                patch_size=config.MODEL.PATCH_SIZE,
                mask_ratio=config.DATA.MASK_RATIO,
            )
        else:
            raise ValueError(f"Unknown dataset {dataset}.")
    else:
        if dataset == "brats":
            return BratsFinetuneData(
                data_dir=config.DATA.DATA_DIR,
                img_size=config.DATA.IMG_SIZE,
                batch_size=config.DATA.BATCH_SIZE,
                num_workers=config.DATA.NUM_WORKERS,
                train_frac=config.DATA.TRAIN_FRAC,
            )
        else:
            raise ValueError(f"Unknown dataset {dataset}.")