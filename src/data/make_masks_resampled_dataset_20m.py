"""
Make masks dataset without 60m resolution bands, 
and with the 10m resolution bands downsample to 20m.

"""

import os
import gc
import cv2
import operator
import rasterio
import logging
import numpy as np

from tqdm import tqdm
from glob import glob
from rasterio.enums import Resampling as resampling

import sys

sys.path.append("../")
sys.path.append("../../")
from config import interim_data_dir

from src.utils import (
    get_img_bands,
    get_safe_dirs,
    date_from_safedir,
    band_from_imgpath,
    read_shapefile,
    mask_raster,
)

logger = logging.Logger(name="data-ingress")

valid_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

res_groups = {
    "60": ["B01", "B09", "B10"],
    "20": ["B05", "B06", "B07", "B8A", "B11", "B12"],
    "10": ["B02", "B03", "B04", "B08"],
}


def safe_create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


from contextlib import contextmanager

import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling

# use context manager so DatasetReader and MemoryFile get cleaned up automatically
@contextmanager
def resample_raster(raster, scale=2):
    t = raster.transform

    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = int(raster.height * scale)
    width = int(raster.width * scale)

    profile = raster.profile
    profile.update(transform=transform, driver="GTiff", height=height, width=width)

    data = raster.read(  # Note changed order of indexes, arrays are band, row, col order not row, col, band
        out_shape=(raster.count, height, width), resampling=Resampling.bilinear
    )

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return


def run(dataset="train"):

    #  Setup directories
    masks_dir = interim_data_dir / "masks_resampled_20m"
    safe_create_dir(masks_dir)

    # train / test dir under masks
    dataset_dir = masks_dir / dataset
    safe_create_dir(dataset_dir)

    # print('Creating {} feature dataset'.format(dataset))

    # print('Reading shapefile...')
    shp_df = read_shapefile(dataset)

    date_dirs = (interim_data_dir / "images-merged").glob("*")

    for date_dir in tqdm(date_dirs, desc="dates"):

        date = date_dir.stem

        # print('Reading image bands...')
        img_fpaths = date_dir.glob("*.jp2")

        for img_fpath in tqdm(img_fpaths, desc="images"):

            # Grab the band from the image path
            band = img_fpath.stem

            if band not in valid_bands:
                # print("Skipping", band)
                continue

            [res_group] = [grp for grp, bands in res_groups.items() if band in bands]

            if res_group == "10":
                with rasterio.open(img_fpath) as raster:
                    with resample_raster(raster, scale=0.5) as raster_resampled:
                        # print('Masking raster...')
                        masks = mask_raster(shp_df.geometry, raster_resampled)

                        # Save mask for each farm in raster
                        for farm_id in masks.keys():

                            farm_dir = dataset_dir / str(farm_id)
                            safe_create_dir(farm_dir)

                            farm_date_dir = farm_dir / date
                            safe_create_dir(farm_date_dir)

                            mask = masks[farm_id]

                            mask_fname = farm_date_dir / f"{band}.npy"

                            np.save(mask_fname, mask)
            else:
                with rasterio.open(img_fpath) as raster:
                    # print('Masking raster...')
                    masks = mask_raster(shp_df.geometry, raster)

                    # Save mask for each farm in raster
                    for farm_id in masks.keys():

                        farm_dir = dataset_dir / str(farm_id)
                        safe_create_dir(farm_dir)

                        farm_date_dir = farm_dir / date
                        safe_create_dir(farm_date_dir)

                        mask = masks[farm_id]

                        mask_fname = farm_date_dir / f"{band}.npy"

                        np.save(mask_fname, mask)

            del masks
            gc.collect()


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    run("train")
    run("test")
