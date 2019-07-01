import os
import gc
import rasterio
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

from glob import glob
from rasterio import mask
from tqdm import tqdm

import sys
sys.path.append('../../')
from config import raw_data_dir, interim_data_dir

logger = logging.Logger(name='data-ingress')


def get_img_bands(safe_dir):
    return glob(f'{safe_dir}/**/IMG_DATA/*.jp2', recursive=True)


def get_safe_dirs():
    return glob(raw_data_dir + '/*.SAFE')


def date_from_safedir(safe_dir):
    date_str: str = os.path.basename(safe_dir).split('.')[0].split('_')[2][:8]

    return str(pd.Timestamp(date_str).date())


def band_from_imgpath(fpath):
    """
    Get the Band from fpath

    :param fpath:
    :return: date, band
    """
    basename: str = os.path.basename(fpath).split('.')[0]
    _, _, band = basename.split('_')
    return band


def read_shapefile(dataset):
    fp = os.path.join(raw_data_dir, '{}/{}.shp'.format(dataset, dataset))
    sh_df = gpd.read_file(fp)

    # Drop NaNs
    sh_df = sh_df.loc[~sh_df.geometry.isna()]

    # Use Field IDs as index
    sh_df.set_index('Field_Id', inplace=True)

    # Convert shp data to correct coordinate system
    sh_df = sh_df.to_crs({'init': 'epsg:32734'})

    return sh_df


def get_shape(geom, raster):
    if geom is None:
        return None
    try:
        out_image, out_transform = mask.mask(raster, [geom], crop=True)
    except ValueError:
        return None
    return out_image


def mask_raster(shapes, raster, return_missing=False):
    """
    Get masks from raster using polygons in shapefile

    :param shapes:
    :param raster: Raster data
    :param return_missing: Flag to return data for masks
                           not found on raster
    :return:
    """

    masks = {}
    for id, shape in shapes.items():
        mask = get_shape(shape, raster)
        if mask is None:
            if return_missing:
                masks[id] = np.ones(shape=(1, 10, 10)) * -1
            else:
                continue

        else:
            masks[id] = mask

    return masks


def check_mask_data(test_mask):
    # TODO: Better deal with multi-band images
    if test_mask.shape[0] > 1:
        logger.warning('Multi-band data might cause kak')


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
