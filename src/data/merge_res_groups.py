"""
What? 

Stack all images in a resolution group into one big ass image.

NOTE! Run `rb_merge_rasters.ipynb` before trying to run this script. 

INPUT: Merged JFP and JEP images in data/interim/images-merges
OUTPUT: All images in a res-group channel-stacked 


"""


import os
import gc

import rasterio
from rasterio.merge import merge
from tqdm import tqdm

from config import interim_data_dir

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

res_groups = {
    "60": ["B01", "B09", "B10"],
    "20": ["B05", "B06", "B07", "B8A", "B11", "B12"],
    "10": [
        "B02",
        "B03",
        "B04",
        "B08",
        #  "TCI"
    ],
}

in_dir = os.path.join(interim_data_dir, "images-merged/")
res_groups_dir = os.path.join(interim_data_dir, "res-groups/")


def merge_res_group(res_group, date):

    out_dir = os.path.join(res_groups_dir, date)
    out_fp = os.path.join(out_dir, f"{res_group}.jp2")

    if os.path.isfile(out_fp):
        return

    print(f"Merging rasters of {res_group}m resolution...")

    bands = res_groups[res_group]
    bands_fpaths = [os.path.join(in_dir, date, f"{band}.jp2") for band in bands]

    # Read meta data from first band
    src0 = rasterio.open(bands_fpaths[0])
    meta = src0.meta.copy()
    src0.close()

    # Update meta data
    meta.update({"count": len(bands)})

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with rasterio.open(out_fp, "w", **meta) as dst:
        for i, fp in enumerate(bands_fpaths, start=1):
            with rasterio.open(fp) as band_raster:
                dst.write_band(i, band_raster.read(1))

    print("done!")
    print("Saved to ", out_fp)

    test_saved_data(out_fp, len(bands))


def test_saved_data(fp, expected_bands):
    with rasterio.open(fp) as raster:
        assert raster.count == expected_bands


if __name__ == "__main__":

    for date in dates[:1]:
        print("-" * 50, date, "-" * 50)
        print("")
        for res_group in res_groups.keys():
            merge_res_group(res_group, date)