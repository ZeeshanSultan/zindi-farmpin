import numpy as np
import pandas as pd

VALID_AGG_METHODS = ["mean", "median", "min", "max", "std"]


def calc_reip(bands_data):
    """

    """
    a = (bands_data['B04'] + bands_data['B07']) / 2

    b = (a - bands_data['B05']) / (bands_data['B06'], bands_data['B05'])

    return 700 + 40 * b



def calc_bri(bands_data):
    """
    (1⁄𝐵𝑎𝑛𝑑3 − 1⁄𝐵𝑎𝑛𝑑5) /𝐵𝑎𝑛𝑑6
    """
    return (1 / bands_data["B03"] - 1 / bands_data["B05"]) / (bands_data["B06"])


def calc_ipvi(bands_data):
    """
    I think it should be band5 + band3 at the end
    0.5 * (𝐵𝑎𝑛𝑑8 / (𝐵𝑎𝑛𝑑8 + 𝐵𝑎𝑛𝑑5)) * (𝐵𝑎𝑛𝑑5 − 𝐵𝑎𝑛𝑑3)/(𝐵𝑎𝑛𝑑5 + 𝐵𝑎𝑛𝑑5) + 1)  
    """

    return (
        0.5
        * (bands_data["B08"] / (bands_data['B08'] + bands_data["B05"]))
        * (1 + (bands_data["B05"] - bands_data["B03"])/(bands_data['B05'] + bands_data['B03']))
    )

def calc_savi(bands_data):
    """
    1.5 * (𝐵𝑎𝑛𝑑8 − 𝐵𝑎𝑛𝑑4) / (𝐵𝑎𝑛𝑑8 + 𝐵𝑎𝑛𝑑4 + 0.5)
    """

    return (
        1.5
        * (bands_data["B08"] - bands_data["B04"])
        / (bands_data["B08"] + bands_data["B04"] + 0.5)
    )


def calc_ndvi(bands_data):
    return (bands_data["B08"] - bands_data["B04"]) / (
        bands_data["B08"] + bands_data["B04"]
    )


def calc_cvi(bands_data):
    """
    (𝐵𝑎𝑛𝑑8 ∗ 𝐵𝑎𝑛𝑑4) / (𝐵𝑎𝑛𝑑3)^2
    """
    return (bands_data["B08"] * bands_data["B04"]) / (bands_data["B03"] ** 2)


def calc_datt1(bands_data):
    """
    𝐵𝑎𝑛𝑑8 − 𝐵𝑎𝑛𝑑5 / 𝐵𝑎𝑛𝑑8 − 𝐵𝑎𝑛𝑑4
    """
    return (bands_data["B08"] - bands_data["B05"]) / (
        bands_data["B08"] - bands_data["B04"]
    )

def calc_datt3(bands_data):
    """
    𝐵𝑎𝑛𝑑8A / 𝐵𝑎𝑛𝑑3 * 𝐵𝑎𝑛𝑑5
    """
    return (bands_data["B8A"]) / (
        bands_data["B03"] * bands_data["B05"]
    )


def calc_evi(bands_data):
    """
    2.5* (𝐵𝑎𝑛𝑑8 − 𝐵𝑎𝑛𝑑4) / (𝐵𝑎𝑛𝑑8 + 6 ∗ 𝐵𝑎𝑛𝑑4 − 7.5 ∗ 𝐵𝑎𝑛𝑑2 + 1)
    """
    return (
        2.5
        * (bands_data["B08"] - bands_data["B04"])
        / (bands_data["B08"] + 6 * bands_data["B04"] - 7.5 * bands_data["B02"] + 1)
    )


def calc_arvi2(bands_data):
    return -0.18 + 1.17 * calc_ndvi(bands_data)


def calc_atsavi(bands_data):
    """
    a ∗ (Band8 − a ∗ Band4 − b) / (Band8 + Band4 − ab + X(1 + 𝑎2))
    
    a = 1.22, b=0.03, X=0.08
    """

    a = 1.22
    b = 0.03
    X = 0.08

    return (
        a
        * (bands_data["B08"] - a * bands_data["B04"] - b)
        / (bands_data["B08"] + bands_data["B04"] - a * b + X * (1 + a ** 2))
    )


def calc_maccioni(bands_data):
    """
    (𝐵𝑎𝑛𝑑7 − 𝐵𝑎𝑛𝑑5) / (𝐵𝑎𝑛𝑑7 − 𝐵𝑎𝑛𝑑4)
    """
    return (bands_data["B07"] - bands_data["B05"]) / (
        bands_data["B07"] - bands_data["B04"]
    )


def calc_arvi(bands_data):
    """
    (𝐵𝑎𝑛𝑑8 − (𝐵𝑎𝑛𝑑4 − 𝛾(𝐵𝑎𝑛𝑑2 −𝐵𝑎𝑛𝑑4)) / 
    (𝐵𝑎𝑛𝑑8 + (𝐵𝑎𝑛𝑑4 − 𝛾(𝐵𝑎𝑛𝑑2 −𝐵𝑎𝑛𝑑4))
    
    𝛾 = 1
    
    """
    nom = bands_data["B08"] - (
        bands_data["B04"] - (bands_data["B02"] - bands_data["B04"])
    )
    denom = bands_data["B08"] + (
        bands_data["B04"] - (bands_data["B02"] - bands_data["B04"])
    )

    return nom / denom


def calc_gari(bands_data):
    """
    𝐵𝑎𝑛𝑑8 − (𝐵𝑎𝑛𝑑3 − (𝐵𝑎𝑛𝑑2 − 𝐵𝑎𝑛𝑑4)) /
    𝐵𝑎𝑛𝑑8 − (𝐵𝑎𝑛𝑑3 + (𝐵𝑎𝑛𝑑2 − 𝐵𝑎𝑛𝑑4))
    
    """
    nom = bands_data["B08"] - (
        bands_data["B03"] - (bands_data["B02"] - bands_data["B04"])
    )
    denom = bands_data["B08"] + (
        bands_data["B03"] + (bands_data["B02"] - bands_data["B04"])
    )

    return nom / denom


def calc_gbndvi(bands_data):
    """
    𝐵𝑎𝑛𝑑8 − (𝐵𝑎𝑛𝑑3 + 𝐵𝑎𝑛𝑑2) /
    𝐵𝑎𝑛𝑑8 + (𝐵𝑎𝑛𝑑3 + 𝐵𝑎𝑛𝑑2)
    """

    nom = bands_data["B08"] - (bands_data["B03"] + bands_data["B02"])
    denom = bands_data["B08"] + (bands_data["B03"] + bands_data["B02"])

    return nom / denom


def calc_mnsi(bands_data):
    """
    0.404 ∗ Band3 + 0.039 ∗ Band4 − 0.505 ∗ Band6 + 0.762 ∗ Band8
    """

    return (
        0.404 * bands_data["B03"]
        + 0.039 * bands_data["B04"]
        - 0.505 * bands_data["B06"]
        + 0.762 * bands_data["B08"]
    )


def calc_msbi(bands_data):
    """
    0.406 ∗ Band3 + 0.600 ∗ Band4 + 0.645 ∗ Band6 + 0.243 ∗ Band8
    """

    return (
        0.406 * bands_data["B03"]
        + 0.6 * bands_data["B04"]
        + 0.645 * bands_data["B06"]
        + 0.762 * bands_data["B08"]
    )


# Just the bands
def calc_band_2(bands_data):
    return bands_data["B02"]


def calc_band_3(bands_data):
    return bands_data["B03"]


def calc_band_4(bands_data):
    return bands_data["B04"]


def calc_band_5(bands_data):
    return bands_data["B05"]


def calc_band_6(bands_data):
    return bands_data["B06"]


def calc_band_7(bands_data):
    return bands_data["B07"]


def calc_band_8(bands_data):
    return bands_data["B08"]


def calc_band_8a(bands_data):
    return bands_data["B8A"]


def calc_band_11(bands_data):
    return bands_data["B11"]


def calc_band_12(bands_data):
    return bands_data["B12"]


def agg_arr(arr, agg_method):
    """
    aggregate an array with a nan-aware numpy method
    """
    assert agg_method in VALID_AGG_METHODS
    return eval(f"np.nan{agg_method}")(arr.flatten())


def calc_vi_ts(farm_data, farm_id, agg_methods, which_vi="ndvi"):
    """
    Calculate vegetation index time series for a farm
    """
    vi_data = {}
    for date, bands_data in farm_data.items():
        # Get aggregated statistics of VI signals for this date
        vi_data[date] = [
            agg_arr(eval(f"calc_{which_vi}")(bands_data), agg_method)
            for agg_method in agg_methods
        ]

    vi = pd.DataFrame.from_dict(vi_data, orient="index")
    vi.index = pd.to_datetime(vi.index)
    vi.index.name = "time"
    vi.columns = [f"{which_vi}_{agg_method}" for agg_method in agg_methods]

    return pd.concat([vi], keys=[farm_id], names=["farm_id"])

