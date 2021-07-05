from common_for_tests import multi_raster_16b
from telluric import GeoRaster2, MutableGeoRaster, GeoVector
from telluric.constants import MERCATOR_RESOLUTION_MAPPING, WGS84_CRS
import telluric as tl
from affine import Affine
import pytest


def test_as_mutable():
    raster = multi_raster_16b().as_mutable()
    assert isinstance(raster, MutableGeoRaster)


def test_construction_mutable_raster():
    raster = MutableGeoRaster.empty_from_roi(GeoVector.from_xyz(300, 300, 13),
                                             resolution=MERCATOR_RESOLUTION_MAPPING[13])
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


def test_pixel_change_in_mutable_raster():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    val = raster.image[1, 3, 3]
    assert val is not 4
    raster.image[1, 3, 3] = 4
    assert raster.image[1, 3, 3] == 4


def test_pixel_change_in_immutable_raster():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=False)
    val = raster.image[1, 300, 300]
    assert val is not 4
    with pytest.raises(ValueError):
        raster.image[1, 300, 300] = 4
    assert raster.image[1, 300, 300] == val


def test_image_setter_in_mutable_raster():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    (indeces, width, height) = raster.shape
    raster.image = raster.image[:, 50:, 50:]
    assert (indeces, width-50, height-50) == raster.shape


def test_image_setter_fail_because_wrong_shape():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    shape = raster.shape
    with pytest.raises(tl.georaster.GeoRaster2Error):
        raster.image = raster.image[0][50:, 50:]
    assert shape == raster.shape


def test_image_setter_in_imutable_raster():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=False)
    with pytest.raises(AttributeError):
        raster.image = raster.image[:, 50:, 50:]


def test_band_names_setter():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    new_band_names = ["1", "2", "3"]
    assert raster.band_names is not new_band_names
    raster.band_names = new_band_names
    assert raster.band_names == new_band_names


def test_band_names_setter_fail():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    new_band_names = ["1", "2"]
    prev_band_names = raster.band_names
    with pytest.raises(tl.georaster.GeoRaster2Error):
        raster.band_names = new_band_names
    assert prev_band_names == raster.band_names


def test_changing_bandnames_and_image():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    new_band_names = ["1"]
    (indeces, width, height) = raster.shape
    raster.set_image(raster.image[0:1, 50:, 50:], new_band_names)
    assert (1, width-50, height-50) == raster.shape
    assert raster.band_names == new_band_names


def test_set_crs():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    assert WGS84_CRS is not raster.footprint().crs
    raster.crs = WGS84_CRS
    assert raster.footprint().crs == WGS84_CRS


def test_set_crs_on_immutable():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=False)
    crs = raster.footprint().crs
    with pytest.raises(AttributeError):
        raster.crs = WGS84_CRS
    assert crs == raster.footprint().crs


def test_set_affine():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    affine = Affine(1, 2, 3, 4, 5, 6)
    raster.affine = affine
    assert raster.affine == affine


def test_set_affine_immutable():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=False)
    affine = Affine(1, 2, 3, 4, 5, 6)
    old_affine = raster.affine
    with pytest.raises(AttributeError):
        raster.affine = affine
    assert raster.affine == old_affine


def test_reporject_of_mutable():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True).reproject(dst_crs=WGS84_CRS)
    assert isinstance(raster, MutableGeoRaster)


def test_image_setter_dtype():
    raster = GeoRaster2.open("tests/data/raster/overlap2.tif", mutable=True)
    raster.image = raster.image.astype('uint16')
    assert raster.dtype == raster.image.dtype


def test_conserved_mutability():
    def _mutable(raster):
        return isinstance(raster, MutableGeoRaster)

    raster = multi_raster_16b()
    assert not _mutable(raster)
    mutable_raster = raster.as_mutable()
    assert _mutable(mutable_raster)

    for r in [raster, mutable_raster]:
        copied = r.copy_with()
        assert _mutable(copied) == _mutable(r)
        resized = r.resize(1.0)
        assert _mutable(resized) == _mutable(r)
