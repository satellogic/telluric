import pytest
import os
from telluric.georaster import (
    GeoRaster2, GeoMultiRaster
)


def raster_for_test():
    filename = os.path.join(os.getcwd(), './tests/data/raster/overlap1.tif')
    return GeoRaster2.open(filename)


def test_multiraster_basics():
    raster = raster_for_test()
    multiraster = GeoMultiRaster([raster_for_test()])
    assert (multiraster.image == raster.image).all()
    assert multiraster.shape == raster.shape
    assert multiraster.band_names == raster.band_names
    assert multiraster.crs == raster.crs
    assert multiraster.affine.almost_equals(raster.affine)


def test_multiraster_pixel_crop():
    raster = raster_for_test()
    multiraster = GeoMultiRaster([raster_for_test()])
    multicrop = multiraster[10:100, 20:200]
    rastercrop = raster[10:100, 20:200]
    assert multicrop.shape == rastercrop.shape
    assert multicrop.band_names == rastercrop.band_names
    assert multicrop.crs == rastercrop.crs
    assert multicrop.affine.almost_equals(rastercrop.affine)


@pytest.mark.xfail(reason="require understanding of the diffs made by vrt")
def test_pending_for_data_research():
    raster = raster_for_test()
    multiraster = GeoMultiRaster([raster_for_test()])
    assert multiraster.footprint() == raster.footprint()
    assert multiraster == raster
    multicrop = multiraster[10:100, 20:200]
    rastercrop = raster[10:100, 20:200]
    assert (multicrop.image == rastercrop.image).all()
    assert multicrop == rastercrop
