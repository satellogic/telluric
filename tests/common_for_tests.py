import numpy as np
from affine import Affine
import copy
import telluric as tl
from telluric.constants import WEB_MERCATOR_CRS


def make_test_raster(value=0, band_names=None, height=3, width=4, dtype=np.uint16,
                     crs=WEB_MERCATOR_CRS, affine=None, image=None):
    band_names = band_names or []
    if affine is None:
        affine = Affine.translation(10, 12) * Affine.scale(1, -1)
    if image is None:
        shape = [len(band_names), height, width]
        array = np.full(shape, value, dtype=dtype)
        mask = np.full(shape, False, dtype=bool)
        image = np.ma.array(data=array, mask=mask)
    raster = tl.GeoRaster2(image=image, affine=affine, crs=crs, band_names=band_names)
    return raster


multi_values_16b = {
    'green': 5500,
    'red': 7700,
    'nir': 10000,
    'blue': 11800
}


def sensor_bands_info():
    return copy.deepcopy({
        'blue': {'min': 400, 'max': 510},
        'green': {'min': 515, 'max': 580},
        'pan': {'min': 400, 'max': 750},
        'red': {'min': 590, 'max': 690},
        'nir': {'min': 750, 'max': 900},
        'HC_450': {'min': 445, 'max': 456},
        'HC_500': {'min': 493, 'max': 510},
        'HC_530': {'min': 521, 'max': 538},
        'HC_550': {'min': 540, 'max': 559},
        'HC_570': {'min': 560, 'max': 579},
        'HC_580': {'min': 572, 'max': 592},
        'HC_595': {'min': 583, 'max': 605},
        'HC_610': {'min': 596, 'max': 619},
        'HC_659': {'min': 646, 'max': 671},
        'HC_670': {'min': 657, 'max': 683},
        'HC_680': {'min': 666, 'max': 693},
        'HC_690': {'min': 676, 'max': 703},
        'HC_700': {'min': 686, 'max': 714},
        'HC_710': {'min': 696, 'max': 724},
        'HC_720': {'min': 705, 'max': 734},
        'HC_730': {'min': 714, 'max': 744},
        'HC_740': {'min': 725, 'max': 755},
        'HC_750': {'min': 735, 'max': 765},
        'HC_760': {'min': 744, 'max': 775},
        'HC_770': {'min': 753, 'max': 785},
        'HC_830': {'min': 809, 'max': 844},
    })


hyper_wavelengths = (450, 500, 530, 550, 570, 580, 595, 610, 670, 680,
                     690, 700, 710, 720, 730, 740, 750, 760, 770, 830)


multi_values_8b = {
    'green': 55,
    'red': 77,
    'nir': 100,
    'blue': 118
}


expected_nir = ['HC_770', 'HC_830']
expected_red = ['HC_670', 'HC_610']
expected_green = ['HC_570', 'HC_550', 'HC_530']
expected_blue = ['HC_450', 'HC_500']

nir_value = 137    # nir values 138 136 avg 137
red_value = 115    # red values 116 114 avg 115
green_value = 106  # green values 104 106 108 avg 106
blue_value = 101   # blue values 100 102 avg 101


def multi_raster_16b():
    source_raster = make_test_raster(4200, ['green', 'red', 'nir', 'blue'], dtype=np.uint16)
    array = source_raster.image.data
    array.setflags(write=1)
    array[0, :, :] = multi_values_16b['green']
    array[1, :, :] = multi_values_16b['red']
    array[2, :, :] = multi_values_16b['nir']
    array[3, :, :] = multi_values_16b['blue']
    return source_raster.copy_with(image=array)


def multi_raster_8b():
    source_raster = make_test_raster(4200, ['green', 'red', 'nir', 'blue'], dtype=np.uint16)
    array = source_raster.image.data
    array.setflags(write=1)
    array[0, :, :] = multi_values_8b['green']
    array[1, :, :] = multi_values_8b['red']
    array[2, :, :] = multi_values_8b['nir']
    array[3, :, :] = multi_values_8b['blue']
    return source_raster.copy_with(image=array)


def multi_raster_with_no_data():
    source_raster = multi_raster_8b()
    source_raster.image.setflags(write=1)
    source_raster.image.mask[0, 1, 2] = True
    source_raster.image.mask[1, 2, 3] = True
    source_raster.image.mask[2, 0, 0] = True
    source_raster.image.mask[3, 1, 3] = True
    source_raster.image.setflags(write=0)
    return source_raster


hyper_bands = ["HC_%i" % wl for wl in hyper_wavelengths]


def hyper_raster():
    source_raster = make_test_raster(142, hyper_bands, dtype=np.uint8)
    source_raster.image.setflags(write=1)
    for i, band in enumerate(hyper_bands):
        source_raster.image.data[i, :, :] = 2 * i + 100
    source_raster.image.setflags(write=0)
    return source_raster


def hyper_raster_with_no_data():
    source_raster = hyper_raster()
    source_raster.image.setflags(write=1)
    source_raster.image.mask[:, 1, 2] = True
    source_raster.image.mask[2, 2, 3] = True
    source_raster.image.mask[5, 0, 0] = True
    source_raster.image.mask[14, 1, 3] = True
    source_raster.image.mask[15, 1, 3] = True
    source_raster.image.setflags(write=0)
    return source_raster
