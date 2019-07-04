import os
import gc
import cv2
import operator
import rasterio
import logging
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')
sys.path.append('../../')
from config import interim_data_dir

from src.utils import get_img_bands, get_safe_dirs, date_from_safedir, band_from_imgpath, read_shapefile, mask_raster

logger = logging.Logger(name='data-ingress')

# Dimensions to zero-pad images to
MAX_DIMS = (100, 100)


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


def zeropad_img(img, shape=MAX_DIMS):
    """
    TODO: Normalize before padding
    TODO: Parameterize anchoring


    :param img:
    :param shape:
    :return:
    """

    # Size of border
    v_border = int(np.ceil((shape[0] - img.shape[0]) / 2))
    h_border = int(np.ceil((shape[1] - img.shape[1]) / 2))

    v_diff = shape[0] - (img.shape[0] + 2 * v_border)
    h_diff = shape[1] - (img.shape[1] + 2 * h_border)

    new_img = cv2.copyMakeBorder(
        img,
        top=v_border, bottom=v_border + v_diff,
        left=h_border, right=h_border + h_diff,
        borderType=cv2.BORDER_CONSTANT, value=0
    )

    assert new_img.shape == shape, 'zero padding issue'

    return new_img


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

    safe_dirs = get_safe_dirs()
    for safe_dir in tqdm(safe_dirs, desc='time'):

        date = date_from_safedir(safe_dir)

        out_dir = os.path.join(dataset_dir, date)
        safe_create_dir(out_dir)

        logger.info('Reading image bands...')
        img_band_fpaths = get_img_bands(safe_dir)
        logger.info(f'\tFound {len(img_band_fpaths)} image bands to process')

        for img_fpath in img_band_fpaths:

            band = band_from_imgpath(img_fpath)

            logger.info('Processing band ', band)
            with rasterio.open(img_fpath) as raster:
                logger.info('Masking raster...')
                masks = mask_raster(shp_df.geometry, raster, return_missing=dataset == 'test')

                # Create zeropadded masks
                logger.info('Zero padding...')
                zp_masks = {id: zeropad_img(mask) for id, mask in masks.items()}

            # Save masks for each farm
            for farm_id in masks.keys():
                mask = masks[farm_id]
                zp_mask = zp_masks[farm_id]

                farm_dir = os.path.join(out_dir, str(farm_id))
                safe_create_dir(farm_dir)

                mask_fname = os.path.join(farm_dir, f'{band}.npy')
                zp_mask_fname = os.path.join(farm_dir, f'{band}_zp.npy')

                np.save(mask_fname, mask)
                np.save(zp_mask_fname, zp_mask)

            del masks
            del zp_masks
            gc.collect()


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    run('train')
    run('test')
