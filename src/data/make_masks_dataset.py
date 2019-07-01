import os
import gc
import rasterio
import logging
import numpy as np
from tqdm import tqdm

from config import interim_data_dir

from src.utils import get_img_bands, get_safe_dirs, date_from_safedir, band_from_imgpath, read_shapefile, mask_raster

logger = logging.Logger(name='data-ingress')


def safe_create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def run(dataset='train'):
    # Setup directories
    masks_dir = os.path.join(interim_data_dir, 'masks')
    safe_create_dir(masks_dir)

    # train / test dir under masks
    dataset_dir = os.path.join(masks_dir, dataset)
    safe_create_dir(dataset_dir)

    logger.info('Creating {} feature dataset'.format(dataset))

    logger.info('Reading shapefile...')
    shp_df = read_shapefile(dataset)

    safe_dirs = get_safe_dirs()
    for safe_dir in tqdm(safe_dirs, desc='time'):

        date = date_from_safedir(safe_dir)

        out_dir = os.path.join(dataset_dir, date)
        safe_create_dir(out_dir)

        logger.info('Reading image bands...')
        img_band_fpaths = get_img_bands(safe_dir)
        logger.info(f'\tFound {len(img_band_fpaths)} image bands to process')

        for img_fpath in tqdm(img_band_fpaths, desc='bands'):
            band = band_from_imgpath(img_fpath)

            logger.info('Processing band ', band)
            with rasterio.open(img_fpath) as raster:
                logger.info('Masking raster...')
                masks = mask_raster(shp_df.geometry, raster, return_missing=dataset == 'test')

            # Save masks for each farm
            for farm_id, mask in masks.items():
                farm_dir = os.path.join(out_dir, str(farm_id))
                safe_create_dir(farm_dir)

                mask_fname = os.path.join(farm_dir, f'{band}.npy')

                np.save(mask_fname, mask)

            del masks
            gc.collect()


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    run('train')
    run('test')
