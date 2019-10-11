import os
import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

import sys

sys.path.append("../")
from config import raw_data_dir, processed_data_dir, interim_data_dir

import rasterio
from rasterio.mask import raster_geometry_mask

from src.utils import read_shapefile

from multiprocessing import Pool, cpu_count

res_groups = {
    "60": ["B01", "B09", "B10"],
    "20": ["B05", "B06", "B07", "B8A", "B11", "B12"],
    "10": ["B02", "B03", "B04", "B08", "TCI"],
}

# Dimensions to zero-pad images to
MAX_DIMS = {
    "10": (90, 90), 
    "20": (50, 50), 
    "60": (20, 20)
}

dates = [
    "2017-01-01",
    "2017-02-10",
    "2017-03-22",
    "2017-06-20",
    "2017-07-15",
    "2017-08-19",
    "2017-01-31",
    "2017-03-12",
    "2017-05-31",
    "2017-07-10",
    "2017-08-04",
]

bands = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
    "TCI",
]


# Input images
images_dir = os.path.join(interim_data_dir, "images-merged")


def get_mask_window(shape, raster, max_dims=None):

    # Get the boolean mask and window of shape
    try:
        bool_mask, transform, bb_window = raster_geometry_mask(
            raster, [shape], crop=True
        )
    except ValueError:
        return None

    if max_dims is None:
        return bb_window
    else:
        # Turn mask into int array with 1 at farm pixels
        int_mask = np.bitwise_not(bool_mask).astype(int)

        # Get the shape of the bounding box window
        bb_shape = (bb_window.height, bb_window.width)

        # Get number of pixels to add to x and y dims
        pad_x = int(np.ceil((max_dims[1] - bb_shape[1]) / 2))
        pad_y = int(np.ceil((max_dims[0] - bb_shape[0]) / 2))

        # Get a window with padding around it and the desired shape in the center
        # Depending on the shape, this window can be a different size than max_dims
        window_ = rasterio.mask.geometry_window(
            raster,
            [shape],
            pad_x=pad_x,
            pad_y=pad_y,
            pixel_precision=2,  # I found this fixes some rounding errors
        )

        # To fix sizing issues, create a new window that starts at the same top left anchor, but of a fixed width and height
        window = rasterio.windows.Window(
            col_off=window_.col_off,
            row_off=window_.row_off,
            width=max_dims[1],
            height=max_dims[0],
        )

        return window


def get_farm_data(masked_raster, raster, farm_shape, out_shape):

    bands = raster.count

    # The window defining where on the raster this shape is found
    mask_window = get_mask_window(farm_shape, raster, max_dims=None)

    if mask_window is None:
        return 0.0

    # The same window, but padded to the desired dims
    mask_window_padded = get_mask_window(farm_shape, raster, max_dims=out_shape)

    # Crop out the image data around the shape
    win_img = raster.read(window=mask_window_padded)[0]

    # Crop out the boolean mask of all farms in the window around the farm of interest
    win_mask_other = masked_raster[
        mask_window_padded.row_off : mask_window_padded.row_off + out_shape[0],
        mask_window_padded.col_off : mask_window_padded.col_off + out_shape[1],
    ]

    temp = np.copy(masked_raster)

    temp[: mask_window.row_off, :] = False
    temp[mask_window.row_off + mask_window.width :, :] = False
    temp[:, : mask_window.col_off] = False
    temp[:, mask_window.col_off + mask_window.height :] = False

    win_mask_farm = temp[
        mask_window_padded.row_off : mask_window_padded.row_off + out_shape[0],
        mask_window_padded.col_off : mask_window_padded.col_off + out_shape[1],
    ]

    del temp

    data = np.zeros((bands + 2, *out_shape))
    data[:bands] = win_img
    data[-2] = win_mask_other
    data[-1] = win_mask_farm

    return data


def get_farms_data(farm_shapes, raster, out_shape):
    """
    
    """

    # Get an boolean mask array with all farms in it
    masked_raster, _, _ = rasterio.mask.raster_geometry_mask(
        raster, farm_shapes.geometry.to_list(), invert=True
    )

    all_data = {}

    for id, shape in tqdm(farm_shapes.iteritems()):
        farm_data = get_farm_data(masked_raster, raster, shape, out_shape)

        if type(farm_data) == np.ndarray:
            all_data[id] = farm_data

    return all_data


def extract_masks(out_dir, dataset, date):

    print("-" * 50, date, "-" * 50)

    # create dir
    dataset_dir = os.path.join(out_dir, dataset)

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    # create dir
    date_dir = os.path.join(dataset_dir, date)

    if not os.path.isdir(date_dir):
        os.makedirs(date_dir)

    shp_df = read_shapefile(dataset)
    farm_shapes = shp_df.geometry

    for band in bands:

        print("\tBand: ", band)
        fp = os.path.join(images_dir, date, f"{band}.jp2")

        [res_group] = [grp for grp, bands in res_groups.items() if band in bands]
        out_shape = MAX_DIMS[res_group]

        with rasterio.open(fp) as raster:
            farms_data = get_farms_data(farm_shapes, raster, out_shape)

        save_data(farms_data, band, date_dir)


def save_data(farms_data, band, dir):

    print("Saving...", end="")
    for farm_id, data in farms_data.items():
        out_fp = os.path.join(dir, f"farm_{farm_id}_{band}.npy")
        np.save(out_fp, data)
    print("done")


def run(dataset):

    print("-" * 50, dataset, "-" * 50)

    for date in dates:

        img_dir = os.path.join(images_dir, date)

        out_dir = os.path.join(processed_data_dir, "data_v1")

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        extract_masks(out_dir, dataset, date)


if __name__ == "__main__":

    run("train")
    run("test")

