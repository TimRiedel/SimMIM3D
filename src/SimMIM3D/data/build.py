from .brats_image_data import BratsImageData
from .brats_data import BratsData

def build_data(config, is_pretrain=True):
    if is_pretrain:
        return BratsImageData(
            data_dir=config.DATA.BRATS_DATA_DIR,
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.PATCH_SIZE,
            mask_ratio=config.DATA.MASK_RATIO,
            batch_size=config.DATA.BATCH_SIZE, 
            num_workers=config.DATA.NUM_WORKERS
        ) 
    else:
        return BratsData(
            data_dir=config.DATA.BRATS_DATA_DIR,
            img_size=config.DATA.IMG_SIZE,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS
        )