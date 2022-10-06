import os
import pytest
import json
import rasterio
import numpy as np
from copy import deepcopy
from tempfile import TemporaryDirectory
from affine import Affine

import telluric as tl
from telluric.constants import WEB_MERCATOR_CRS

from telluric.util.raster_utils import (_calc_overviews_factors, _has_internal_perdataset_mask,
                                        _mask_from_masked_array, convert_to_cog)


base_affine = Affine.translation(20, -20) * Affine.scale(2, -2)


def sample_raster_image(data=None, mask=None, bands=3, height=4, width=5):
    if data is None:
        data = np.random.rand(bands * height * width) * 10
        data = data.reshape(bands, height, width)
        data = np.floor(data).astype('uint8')

    if mask is None:
        mask = np.repeat((data == 0).all(axis=0, keepdims=True).copy(), bands, 0)

    return np.ma.masked_array(data=data, mask=mask)


def save_raster_image(file_name, image, affine=base_affine, crs=WEB_MERCATOR_CRS):
    count, height, width = image.shape
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(file_name, 'w', count=count, width=width,
                           height=height, driver='GTiff', dtype=image.dtype,
                           crs=crs, transform=affine) as raster:

            raster.write(image)


def test_cog_simple_file():
    with TemporaryDirectory() as dir_name:
        file_name = os.path.join(dir_name, 'temp.tif')
        image = sample_raster_image()
        tl.GeoRaster2(deepcopy(image), crs=WEB_MERCATOR_CRS,
                      affine=base_affine).save(file_name)
        with rasterio.open(file_name) as raster:
            data = raster.read()
            assert (image.data == data).all()
            mask = raster.read_masks()
            expected_mask = _mask_from_masked_array(image)
            assert (expected_mask == mask).all()
            assert _has_internal_perdataset_mask(raster)


def test_cog_overviews():
    with TemporaryDirectory() as dir_name:
        source = os.path.join(dir_name, 'source.tif')
        dest = os.path.join(dir_name, 'dest.tif')
        image = sample_raster_image(height=800, width=900)
        tl.GeoRaster2(deepcopy(image), crs=WEB_MERCATOR_CRS,
                      affine=base_affine).save(source)
        convert_to_cog(source, dest)
        assert (os.path.exists(dest))
        assert not os.path.exists("%s.msk" % dest)
        with rasterio.open(dest) as raster:
            data = raster.read()
            assert (image.data == data).all()
            mask = raster.read_masks()
            expected_mask = _mask_from_masked_array(image)
            assert (expected_mask == mask).all()

            # to confirtm that creating a cog remove namespace tags
            assert raster.tags(ns='rio_overview').get('resampling', 'empty') == 'empty'
            for i in raster.indexes:
                assert raster.overviews(i) == _calc_overviews_factors(raster)

            assert _has_internal_perdataset_mask(raster)


def test_cog_move_telluric_tags_to_general_tags_space():
    with TemporaryDirectory() as dir_name:
        source = os.path.join(dir_name, 'source.tif')
        dest = os.path.join(dir_name, 'dest.tif')
        image = sample_raster_image(height=800, width=900)
        band_names = ['red', 'green', 'blue']
        tl.GeoRaster2(deepcopy(image), crs=WEB_MERCATOR_CRS,
                      affine=base_affine, band_names=band_names).save(source)

        convert_to_cog(source, dest)
        tags = tl.GeoRaster2.tags(dest)
        assert (json.loads(tags['telluric_band_names']) == band_names)
        raster = tl.GeoRaster2.open(dest)
        assert raster.band_names == band_names


@pytest.mark.parametrize('height, factors', [
                         (800, [2, 4]),
                         (1024, [2, 4]),
                         (8000, [2, 4, 8, 16, 32])
                         ])
def test_cog_calc_overviews_factors(height, factors):
    image = sample_raster_image(height=height, width=900)
    raster = tl.GeoRaster2(image, crs=WEB_MERCATOR_CRS, affine=base_affine)

    assert (_calc_overviews_factors(raster) == factors)


def test_cog_mask_from_masked_array():
    some_array = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    mask1 = np.array([[False, False, False], [False, False, True]], dtype=bool)
    mask2 = np.array([[False, False, True], [False, False, True]], dtype=bool)
    mask3 = np.array([[False, True, False], [False, False, True]], dtype=bool)
    masks = np.array([mask1, mask2, mask3])
    data = np.array([some_array, some_array, some_array])

    image = np.ma.masked_array(data=data, mask=masks)

    expected_mask = np.array([[255, 0, 0], [255, 255, 0]], dtype=np.uint8)
    assert ((_mask_from_masked_array(image) == expected_mask).all())
