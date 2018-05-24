import os
import rasterio
import numpy as np
from copy import deepcopy
from tempfile import TemporaryDirectory
from affine import Affine

import telluric as tl
from telluric.constants import WEB_MERCATOR_CRS

from telluric.util.raster_utils import (_calc_overviews_factors, _has_mask,
                                   _mask_from_masked_array, convert_to_cog)


base_affine = Affine.translation(20, -20) * Affine.scale(2, -2)


def sample_raster_image(data=None, mask=None, bands=3, height=4, width=5):
    if data is None:
        data = np.random.rand(bands * height * width) * 10
        data = data.reshape(bands, height, width)
        data = np.floor(data).astype('uint8')

    if mask is None:
        mask = (data == 0).copy()

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
            mask = raster.read_mask()
            expected_mask = _mask_from_masked_array(image)
            assert (expected_mask == mask).all()
            assert _has_mask(raster)


def test_cog_overviews():
    with TemporaryDirectory() as dir_name:
        source = os.path.join(dir_name, 'source.tif')
        dest = os.path.join(dir_name, 'dest.tif')
        image = sample_raster_image(height=800, width=900)
        tl.GeoRaster2(deepcopy(image), crs=WEB_MERCATOR_CRS,
                      affine=base_affine).save(source)
        convert_to_cog(source, dest)
        assert(os.path.exists(dest))
        assert not os.path.exists("%s.msk" % dest)
        with rasterio.open(dest) as raster:
            data = raster.read()
            assert (image.data == data).all()
            mask = raster.read_mask()
            expected_mask = _mask_from_masked_array(image)
            assert (expected_mask == mask).all()

            assert raster.tags(ns='rio_overview').get('resampling', 'empty') == 'empty'
            for i in raster.indexes:
                assert raster.overviews(i) == _calc_overviews_factors(raster)

            assert _has_mask(raster)
