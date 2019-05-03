import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

from glob import glob
from rasterio import mask

from config import raw_data_dir, root_dir
from sentinelhub import AwsProductRequest


def get_img_bands(safe_dir):
    f_base: str = os.path.basename(safe_dir).split('.')[0]

    request = AwsProductRequest(data_folder=raw_data_dir,
                                product_id=f_base,
                                tile_list=None,
                                bands=None,
                                # metafiles=['metadata', 'preview/B02', 'qi/MSK_CLOUDS_B00'],
                                safe_format=True)

    # returns a list of requested file locations in the same order as obtained data
    band_filenames = [os.path.join(raw_data_dir, f) for f in request.get_filename_list()]
    return band_filenames


def get_safe_dirs():
    return glob(raw_data_dir + '/*.SAFE')


def read_shapefile(fp):
    sh_df = gpd.read_file(fp)

    # Drop NaNs
    sh_df = sh_df.loc[~sh_df.geometry.isna()]
    sh_df.reset_index(drop=True, inplace=True)

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


def reduce_img(img: np.array):
    assert img.shape[0] == 1, 'Multi-band data might cause kak'

    _, height, width = img.shape

    img_flat = img.flatten()

    return height, width, height * width, img_flat.mean(), img_flat.std()


def to_dataframe(stats):
    data = np.array(stats).reshape(-1, 5)
    return pd.DataFrame(data, columns=['width', 'height', 'size', 'mean', 'std'])


def save_dataframe(df, f_band):
    out_dir = os.path.join(root_dir, 'data/processed/baseline')
    f_base = os.path.basename(f_band).split('.')[0]

    f_out = os.path.join(out_dir, f_base + '.csv')

    print('Saving to ', f_out)

    df.to_csv(f_out)


def main():
    dataset = 'train'

    print('Reading shapefile...', end='')
    shp_df = read_shapefile(os.path.join(raw_data_dir, '{}/{}.shp'.format(dataset, dataset)))
    print('done.')

    safe_dirs = get_safe_dirs()
    safe_dir = safe_dirs[0]

    img_band_fpaths = get_img_bands(safe_dir)
    print(f'Found {len(img_band_fpaths)} image bands to process')

    for img_fpath in img_band_fpaths:
        print('Processing band ', img_fpath)
        with rasterio.open(img_fpath) as raster:
            # Get masks from raster using polygons in shapefile
            masks = [get_shape(g, raster) for g in shp_df.geometry]

            # TODO clean this up
            masks = [m for m in masks if m is not None]

            print(f'{len(masks)} farms successfully masked')

            # Calculate descriptive stats for each mask
            mask_data = [reduce_img(img) for img in masks]

            mask_df = to_dataframe(mask_data)

            save_dataframe(mask_df, f_band=img_fpath)

        break


if __name__ == '__main__':
    # TODO: Get center point Lat Lon as feature
    main()
