import os
import numpy as np
import pandas as pd
import geopandas as gpd

from glob import glob
from tqdm import tqdm
from rasterio import mask

from config import raw_data_dir


def make_sub(probs):
    test_sub = pd.read_csv(os.path.join(raw_data_dir, 'sample_submission.csv'), index_col=0)
    probs_df = pd.DataFrame(probs, columns=list(test_sub))
    probs_df.index = test_sub.index.values
    return probs_df


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

    if out_image.ndim == 3:
        # remove the first dimension
        return out_image[0, :, :]
    elif out_image.ndim == 2:
        return out_image
    else:
        raise ValueError('Mask has unexpected shape: {}'.format(out_image.shape))


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
                masks[id] = np.ones(shape=(10, 10)) * -1
            else:
                continue
        else:
            assert mask.ndim == 2, 'Farm {} masking error'.format(id)

            masks[id] = mask

    return masks


if __name__ == '__main__':
    pass
