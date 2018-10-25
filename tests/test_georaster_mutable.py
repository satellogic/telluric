from common_for_tests import multi_raster_16b
from telluric import GeoRaster2, MutableGeoRaster, GeoVector
from telluric.constants import MERCATOR_RESOLUTION_MAPPING



def test_as_mutable():
    raster = multi_raster_16b().as_mutable()
    assert isinstance(raster, MutableGeoRaster)

def test_construction_mutable_raster():
    raster = MutableGeoRaster.empty_from_roi(GeoVector.from_xyz(300,300,13), resolution=MERCATOR_RESOLUTION_MAPPING[13])
    assert isinstance(raster, MutableGeoRaster)

def test_open_mutable_raster():
    raster = MutableGeoRaster.open("tests/data/raster/overlap2.tif")
    assert isinstance(raster, MutableGeoRaster)

def test_open_mutable_raster_from_georaster():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    assert isinstance(raster, MutableGeoRaster)

def test_open_imutable_raster_from_georaster():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif")
    assert isinstance(raster, GeoRaster2)

