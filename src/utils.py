import os
import gc
import pickle
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

from glob import glob
from tqdm import tqdm

from rasterio import mask

from config import raw_data_dir, interim_data_dir


def make_sub(probs):
    test_sub = pd.read_csv(os.path.join(raw_data_dir, 'sample_submission.csv'), index_col=0)
    probs_df = pd.DataFrame(probs, columns=list(test_sub))
    probs_df.index = test_sub.index.values
    return probs_df


def safe_create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


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


def get_mask(shape, raster):
    if shape is None:
        return None
    try:
        mask_img, mask_transform = mask.mask(raster, [shape], crop=True)
    except ValueError:
        return None

    if mask_img.ndim == 3:
        # remove the first dimension
        return mask_img[0, :, :]
    elif mask_img.ndim == 2:
        return mask_img
    else:
        raise ValueError('Mask has unexpected shape: {}'.format(mask_img.shape))


def mask_raster(shapes, raster, return_missing=False):
    """
    Get masks from raster using polygons in shapefile

    :param shapes:
    :param raster: Raster data
    :param return_missing: Flag to return data for masks
                           not found on raster
    :return:
    """

    assert type(shapes) == gpd.GeoSeries
    assert shapes.name == 'geometry'

    masks = {}
    for id, shape in shapes.items():
        mask = get_mask(shape, raster)

        if mask is None:
            if return_missing:
                masks[id] = np.ones(shape=(10, 10)) * -1
            else:
                continue
        else:
            assert mask.ndim == 2, 'Farm {} masking error'.format(id)

            masks[id] = mask

    return masks


def get_farm_ids(dataset='train', label='JFP'):
    """
    Gets a list of farm ids present in the
    selected dataset and label
    """

    def create_farm_ids(output_fpath):
        """
        Creates a pickle dump of farm ids
        present in datasets (train/test) and
        labels (JEP & JFP)
        """

        farm_ids = {'train': {'JEP': [], 'JFP': []}, 'test': {'JEP': [], 'JFP': []}}

        for dataset in ['train', 'test']:
            shp_df = read_shapefile(dataset)

            for label in ['JEP', 'JFP']:

                # Select an image to load
                img_fpath = os.path.join(interim_data_dir, f'images/2017-01-01/B02_{label}.jp2')

                if not os.path.isfile(img_fpath):
                    raise FileNotFoundError('''
                    Images have not been reordered into interim data dir.
                    \n\nSee readme to run invoke command for re-odering image data. 
                    ''')

                with rasterio.open(img_fpath) as raster:
                    masks = mask_raster(shp_df.geometry, raster, return_missing=False)

                farm_ids[dataset][label] = list(masks.keys())

                del masks
                gc.collect()

        with open(output_fpath, 'wb') as f:
            pickle.dump(farm_ids, f)

        return farm_ids

    ids_fpath = os.path.join(interim_data_dir, 'farm_ids.pkl')

    if not os.path.isfile(ids_fpath):
        farm_ids = create_farm_ids(ids_fpath)

    else:
        with open(ids_fpath, 'rb') as f:
            farm_ids = pickle.load(f)

    return farm_ids[dataset][label]


if __name__ == '__main__':
    pass
