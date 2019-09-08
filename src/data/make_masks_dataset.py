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


def upsample_raster(raster):
    return raster.read(
        out_shape=(raster.height * 2, raster.width * 2, raster.count),
        resampling=resampling.bilinear,
    )


def run(dataset="train"):

    #  Setup directories
    masks_dir = interim_data_dir / "masks_v3"
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
