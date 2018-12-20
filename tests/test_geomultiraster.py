import pytest
import os
from telluric.georaster import (
    GeoRaster2, GeoMultiRaster, GeoRaster2Error
)


def raster_for_test():
    filename = os.path.join(os.getcwd(), './tests/data/raster/overlap1.tif')
    return GeoRaster2.open(filename)


def raster_for_test_2():
    filename = os.path.join(os.getcwd(), './tests/data/raster/overlap2.tif')
    return GeoRaster2.open(filename)


def test_multiraster_basics():
    raster = raster_for_test()
    multiraster = GeoMultiRaster([raster_for_test()])
    assert (multiraster.image == raster.image).all()
    assert multiraster.shape == raster.shape
    assert multiraster.band_names == raster.band_names
    assert multiraster.crs == raster.crs
    assert multiraster.affine.almost_equals(raster.affine)


def test_to_assets():
    raster = raster_for_test()
    multiraster = GeoMultiRaster([raster_for_test()])
    assert raster.to_assets() == multiraster.to_assets()


def test_from_assets_single_raster():
    assets = raster_for_test().to_assets()
    assert GeoMultiRaster.from_assets(assets) == GeoRaster2.from_assets(assets)
    assert isinstance(GeoMultiRaster.from_assets(assets), GeoRaster2)
    assert isinstance(GeoRaster2.from_assets(assets), GeoRaster2)


def test_from_assets_two_rasters():
    assets = GeoMultiRaster([raster_for_test(), raster_for_test()]).to_assets()
    assert GeoMultiRaster.from_assets(assets) == GeoRaster2.from_assets(assets)
    assert isinstance(GeoMultiRaster.from_assets(assets), GeoMultiRaster)
    assert isinstance(GeoRaster2.from_assets(assets), GeoMultiRaster)


def test_multiraster_pixel_crop():
    raster = raster_for_test()
    multiraster = GeoMultiRaster([raster_for_test()])
    multicrop = multiraster[10:100, 20:200]
    rastercrop = raster[10:100, 20:200]
    assert multicrop.shape == rastercrop.shape
    assert multicrop.band_names == rastercrop.band_names
    assert multicrop.crs == rastercrop.crs
    assert multicrop.affine.almost_equals(rastercrop.affine)


def test_compare_multiraster_and_vrt():
    raster = GeoRaster2.from_rasters([raster_for_test()])
    multiraster = GeoMultiRaster([raster_for_test()])
    assert multiraster == raster


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


def test_raises_on_empty_rasters():
    with pytest.raises(GeoRaster2Error):
        GeoMultiRaster([])
