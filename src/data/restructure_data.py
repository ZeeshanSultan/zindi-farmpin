"""
After unzipping the data into their .SAFE file structures,
the layout is confusing and over the top complex.

In this script, simplify the folder structure by creating
folder names of the dump date (eg 2017-01-01) and then copying
all the images (all bands of JEP and JFP) into that folder.

"""

import os
import shutil

from config import interim_data_dir
from src.utils import date_from_safedir, get_safe_dirs, \
    safe_create_dir, get_img_bands, band_from_imgpath


def run():
    safe_dirs = get_safe_dirs()

    if not safe_dirs:
        raise FileNotFoundError('Cannot find any .SAFE dirs. '
                                'Run `invoke download-satellite-data` to download and extract the dataset')

    out_dir = os.path.join(interim_data_dir, 'images')
    safe_create_dir(out_dir)

    for safe_dir in safe_dirs:

        date = date_from_safedir(safe_dir)

        date_dir = os.path.join(out_dir, date)
        safe_create_dir(date_dir)

        label = 'JFP' if 'JFP' in safe_dir else 'JEP'

        img_fpaths = get_img_bands(safe_dir)

        for img_fpath in img_fpaths:
            band = band_from_imgpath(img_fpath)

            out_fname = os.path.join(date_dir, f'{band}_{label}.jp2')

            shutil.copy(img_fpath, out_fname)


if __name__ == '__main__':
    run()
