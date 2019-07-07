"""
Organize the data into a typical structure for deep learning:

[INNER FOLDER STRUCTURE]
{train/test}
    -- {class}
        -- {farm_id}.npy



Because we have a temporal dimension and multiple resolution groups,
we will create such a folder for each group and each time-stamp:

[OUTER FOLDER STRUCTURE]
{res_group}
    -- {train/test}
        -- {class}
            -- {farm_id}_{date}.npy

"""
import os
import cv2
import rasterio
import numpy as np

from scipy.misc import imsave

from config import interim_data_dir, processed_data_dir
from src.utils import read_shapefile, get_farm_ids, mask_raster

dates = ['2017-01-01', '2017-01-31', '2017-02-10',
         '2017-03-12', '2017-03-22', '2017-05-31',
         '2017-06-20', '2017-07-10', '2017-07-15',
         '2017-08-04', '2017-08-19']

res_groups = {
    # '60': ['B01', 'B09', 'B10'],
    # '20': ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],
    '10': ['B02', 'B03', 'B04']  # , 'B08']
}

# Dimensions to zero-pad images to
MAX_DIMS = {
    '10': (90, 90),
    '20': (50, 50),
    '60': (20, 20)
}

classes = list(map(str, np.arange(1, 10)))

output_dir = os.path.join(interim_data_dir, 'images')



def load_img(date, band, label, images_dir):
    fpath = os.path.join(images_dir, date, f'{band}_{label}.jp2')

    with rasterio.open(fpath) as raster:
        img = raster

    return img


def zero_pad_mask(img, shape):
    """
    TODO: Normalize before padding
    TODO: Parameterize anchoring


    :param img:
    :param shape:
    :return:
    """
    # Size of border
    v_border = int(np.ceil((shape[0] - img.shape[0]) / 2))
    h_border = int(np.ceil((shape[1] - img.shape[1]) / 2))

    v_diff = shape[0] - (img.shape[0] + 2 * v_border)
    h_diff = shape[1] - (img.shape[1] + 2 * h_border)

    new_img = cv2.copyMakeBorder(
        img,
        top=v_border, bottom=v_border + v_diff,
        left=h_border, right=h_border + h_diff,
        borderType=cv2.BORDER_CONSTANT, value=0
    )

    assert new_img.shape == shape, 'zero padding issue'

    return new_img


def save_arr(arr, fpath):
    assert type(arr) == np.ndarray, 'expected ndarray, got ' + str(type(arr))
    with open(fpath, 'wb') as f:
        np.save(f, arr)


def save_img(img, fpath):
    imsave(fpath, img)


def extract_data(dataset, label):
    """

    :param dataset:
    :param label:
    :return:
    """
    farm_ids = get_farm_ids(dataset, label)

    shp_df = read_shapefile(dataset)
    shp_df = shp_df.loc[farm_ids]

    for res_group, bands in res_groups.items():

        print('Res group: ', res_group)

        for date in dates:

            print('\t{} ...'.format(date), end='')

            bands_data = {band: {} for band in bands}

            for band in bands:
                img_fpath = os.path.join(output_dir, date, f'{band}_{label}.jp2')

                with rasterio.open(img_fpath) as raster:
                    masks = mask_raster(shp_df['geometry'], raster, return_missing=dataset == 'test')

                zp_masks = {id: zero_pad_mask(mask, shape=MAX_DIMS[res_group]) for id, mask in
                            masks.items()}  # {farm_id: np.array, farm_id: np.array, ... }

                bands_data[band].update(zp_masks)

                del masks
                del zp_masks

            for farm_id in farm_ids:
                # Create empty array for stacked bands data
                stacked_img = np.zeros(shape=(MAX_DIMS[res_group][0], MAX_DIMS[res_group][1], len(bands)))

                for i, (band, masks) in enumerate(bands_data.items()):
                    stacked_img[:, :, i] = masks[farm_id]

                # For training set, add class label to output path
                if dataset == 'train':
                    farm_id_class = shp_df.loc[farm_id]['Crop_Id_Ne']

                    out_img_fpath = os.path.join(processed_data_dir, 'stacked_images',
                                                 f'res_{res_group}', dataset, farm_id_class, f'{farm_id}.jpg')
                else:
                    out_img_fpath = os.path.join(processed_data_dir, 'stacked_images',
                                                 f'res_{res_group}', dataset, f'{farm_id}.jpg')

                # Save the stacked image to disk
                # save_arr(stacked_img, out_img_fpath)
                save_img(stacked_img, out_img_fpath)

            print('done')


def setup_dirs():
    from utils import safe_create_dir

    stacked_images_dir = os.path.join(interim_data_dir, 'stacked_images')

    for res_group in res_groups.keys():
        res_dir = os.path.join(stacked_images_dir, f'res_{res_group}')

        safe_create_dir(res_dir)

        for dataset in ['train', 'test']:
            d = os.path.join(res_dir, dataset)
            safe_create_dir(d)

            if dataset == 'train':
                for c in classes:
                    c_dir = os.path.join(d, c)
                    safe_create_dir(c_dir)


def run():
    setup_dirs()

    if not os.path.isdir(output_dir):
        raise FileNotFoundError('''
                Images have not been reordered into interim data dir.
                 \n\nSee readme to run invoke command for re-ordering image data. 
                 ''')

    for dataset in ['train', 'test']:
        print('\n')
        print('-' * 50 + '  ' + dataset + '  ' + '-' * 50)

        for label in ['JFP', 'JEP']:
            print('-' * 50 + '  ' + label + '  ' + '-' * 50)

            extract_data(dataset, label)


if __name__ == '__main__':
    run()
