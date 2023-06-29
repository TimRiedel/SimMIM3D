from .adni_image_data import AdniData
from .brats_image_data import BratsImageData
from .brats_data import BratsData

def build_data(config, dataset, is_pretrain=True):
    if is_pretrain:
        if dataset == "brats":
            return BratsImageData(
                data_dir=config.DATA.DATA_DIR,
                img_size=config.DATA.IMG_SIZE,
                batch_size=config.DATA.BATCH_SIZE, 
                num_workers=config.DATA.NUM_WORKERS,
                patch_size=config.MODEL.PATCH_SIZE,
                mask_ratio=config.DATA.MASK_RATIO,
            ) 
        elif dataset == "adni":
            return AdniData(
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
            return BratsData(
                data_dir=config.DATA.DATA_DIR,
                img_size=config.DATA.IMG_SIZE,
                batch_size=config.DATA.BATCH_SIZE,
                num_workers=config.DATA.NUM_WORKERS
            )
        else:
            raise ValueError(f"Unknown dataset {dataset}.")