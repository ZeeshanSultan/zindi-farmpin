import os
import gc
import cv2
import operator
import rasterio
import logging
import numpy as np

from tqdm import tqdm
from glob import glob

import sys

sys.path.append('../')
sys.path.append('../../')
from config import interim_data_dir

from src.utils import get_img_bands, get_safe_dirs, date_from_safedir, band_from_imgpath, read_shapefile, mask_raster

logger = logging.Logger(name='data-ingress')

# Dimensions to zero-pad images to
MAX_DIMS = {
    '10': (90, 90),
    '20': (50, 50),
    '60': (20, 20)
}

def get_largest_dims(masks):
    """
    Get the largest width and height from all farms
    TODO: Make this better
    """
    # First find the largest farm
    dims = [mask.shape for mask in masks.values()]

    max_width = max([sh[0] for sh in dims])
    max_height = max([sh[-1] for sh in dims])

    del dims
    gc.collect()

    return max_width, max_height


def safe_create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def run(dataset='train'):

    #  Setup directories
    masks_dir = os.path.join(interim_data_dir, 'masks')
    safe_create_dir(masks_dir)

    # train / test dir under masks
    dataset_dir = os.path.join(masks_dir, dataset)
    safe_create_dir(dataset_dir)

    logger.info('Creating {} feature dataset'.format(dataset))

    logger.info('Reading shapefile...')
    shp_df = read_shapefile(dataset)

    img_dirs = glob(os.path.join(interim_data_dir, 'images/*'))

    for img_dir in tqdm(img_dirs, desc='dates'):

        date = os.path.basename(img_dir)

        date_dir = os.path.join(dataset_dir, date)
        safe_create_dir(date_dir)

        logger.info('Reading image bands...')
        img_fpaths = glob(os.path.join(img_dir, '*.jp2'))

        for img_fpath in tqdm(img_fpaths, desc='images'):

            # Grab the band from the image path
            band = os.path.basename(img_fpath).split('_')[0]

            with rasterio.open(img_fpath) as raster:
                logger.info('Masking raster...')
                masks = mask_raster(shp_df.geometry, raster, return_missing=dataset == 'test')

                # Save mask for each farm in raster
                for farm_id in masks.keys():
                    mask = masks[farm_id]

                    mask_fname = os.path.join(date_dir, f'farm_{farm_id}_{band}.npy')

                    np.save(mask_fname, mask)

                del masks
                gc.collect()


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    run('train')
    run('test')
