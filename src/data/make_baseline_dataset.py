import os
import rasterio
import warnings
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

from glob import glob
from rasterio import mask
from tqdm import tqdm
from scipy.misc import imresize

from config import raw_data_dir, processed_data_dir

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


def impute_missing_mask_features(masks, shp_df):
    """
    Impute data for missing / OOB masks
    using mean values for that class

    :param masks:
    :param shp_df:
    :return:
    """

    # Calculate mean per class
    pass


def check_mask_data(test_mask):
    # TODO: Better deal with multi-band images
    if test_mask.shape[0] > 1:
        logger.warning('Multi-band data might cause kak')


def reduce_img(img: np.array):
    """
    TODO: In future we should only pass 2d arrays here (currently 3d)

    :param img: 3D ndarray
    :return: features
    """

    _, height, width = img.shape

    img_flat = img.flatten()

    return [height, width, height * width, img_flat.mean(), img_flat.std()]


def calculate_ndvi(input_dict):
    """
    ndvi = (b8a - b4) / (b8a + b4)

    :param input_dict:
    :return:
    """
    keys = input_dict['B04'].keys()
    assert keys == input_dict['B8A'].keys(), 'NDVI calculation error - bands dont match'

    ndvi_masks = {}
    # For each mask (shape)
    for key in keys:
        b4 = input_dict['B04'][key]
        b8a = input_dict['B8A'][key]

        # Select first spectral band
        if b4.ndim > 2:
            b4 = b4[0]

        if b8a.ndim > 2:
            b8a = b8a[0]

        # resize b8a to size of b4
        b4 = imresize(b4, b8a.shape)

        ndvi = (b8a - b4) / (b8a + b4)

        # TODO - remove this when bands are sorted out
        ndvi = np.expand_dims(ndvi, axis=0)

        ndvi_masks[key] = ndvi

    # Calculate descriptive features
    ndvi_features = {key: reduce_img(ndvi_img) for key, ndvi_img in ndvi_masks.items()}

    ndvi_df = features_dict_to_dataframe(ndvi_features, band='NDVI')

    return ndvi_df


def features_dict_to_dataframe(mask_data_dict, band, feature_names=['width', 'height', 'size', 'mean', 'std']):
    df = pd.DataFrame.from_dict(mask_data_dict, orient='index')

    assert df.shape[1] == len(feature_names), \
        'List of features given does not match data shape. {} features given, but {} columns in data'.format(
            len(feature_names), df.shape[1]
        )

    # Version with date in feature names
    # df.columns = [f'{band}_{date}_{f}' for f in feature_names]
    df.columns = [f'{band}_{f}' for f in feature_names]

    df.index.name = 'Field_Id'

    return df


def save_dataframe(df, dataset, name='data.csv', ):
    out_dir = os.path.join(processed_data_dir, f'baseline_ndvi/{dataset}')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if '.csv' not in name:
        name = name + '.csv'

    fpath_out = os.path.join(out_dir, name)

    logger.info('Saving to ', fpath_out)

    df.to_csv(fpath_out)


def run(dataset='train'):
    logger.info('Creating {} feature dataset'.format(dataset))

    logger.info('Reading shapefile...')
    shp_df = read_shapefile(dataset)

    safe_dirs = get_safe_dirs()

    for safe_dir in tqdm(safe_dirs, desc='time'):

        date = date_from_safedir(safe_dir)

        logger.info('Reading image bands...')
        img_band_fpaths = get_img_bands(safe_dir)
        logger.info(f'\tFound {len(img_band_fpaths)} image bands to process')

        mask_dfs = []
        ndvi_input_data = {}  # Keep bands 8a and 4
        for img_fpath in tqdm(img_band_fpaths, desc='bands'):
            band = band_from_imgpath(img_fpath)

            logger.info('Processing band ', band)
            with rasterio.open(img_fpath) as raster:
                logger.info('Masking raster...')
                masks = mask_raster(shp_df.geometry, raster, return_missing=dataset == 'test')

            # Save for NDVI calculation
            if band in ['B8A', 'B04']:
                ndvi_input_data[band] = masks

            logger.info(f'{len(masks)} farms successfully masked')

            # TODO
            # check_mask_data(masks[list(masks.keys())[0]])

            # Calculate descriptive stats for each mask
            mask_data = {idx: reduce_img(img) for idx, img in masks.items()}

            # TODO
            # mask_data = impute_missing_masks(mask_data, shp_df)

            mask_df = features_dict_to_dataframe(mask_data, band)

            mask_dfs.append(mask_df)

        # Calculate NDVI
        ndvi_df = calculate_ndvi(ndvi_input_data)

        # Combine all features for all bands
        df = pd.concat(mask_dfs, axis=1)

        df = df.merge(ndvi_df, left_index=True, right_index=True, how='left')

        if dataset == 'train':
            df = df.merge(shp_df.Crop_Id_Ne, left_index=True, right_on='Field_Id', how='left')

        f_out = f'{date}.csv'
        save_dataframe(df, dataset=dataset, name=f_out)
        logger.info(f'Successfully saved feature dataset to {f_out}')


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    run('train')
    run('test')
