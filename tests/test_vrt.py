import os
import pytest
import rasterio
import numpy as np
from packaging import version
from rasterio.io import MemoryFile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from telluric.util.raster_utils import build_vrt
from telluric import GeoFeature, GeoRaster2, constants, FeatureCollection
from telluric.vrt import wms_vrt, boundless_vrt_doc, raster_list_vrt, raster_collection_vrt


record = {
    "type": "Feature",
    "properties": {"bla": "bla"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [
                    34.32128906249999,
                    30.93050081760779
                ],
                [
                    35.9527587890625,
                    30.93050081760779
                ],
                [
                    35.9527587890625,
                    32.879587173066305
                ],
                [
                    34.32128906249999,
                    32.879587173066305
                ],
                [
                    34.32128906249999,
                    30.93050081760779
                ]
            ]
        ]
    }
}


@pytest.mark.skipif(
    version.parse(rasterio.__version__) >= version.parse('1.3'),
    reason="rasterio >= 1.3 produces different values",
)
def test_wms_vrt():
    vector = GeoFeature.from_record(record, crs=constants.WGS84_CRS).geometry
    doc = str(wms_vrt("tests/data/google.xml",
                      bounds=vector,
                      resolution=1).tostring())
    with open("tests/data/raster/google_israel.vrt", 'r') as expected_src:
        expected = expected_src.read()
        expected = expected.replace("--root-folder--", os.getcwd())
        assert expected == doc


def test_georaster_wms_vrt():
    vector = GeoFeature.from_record(record, crs=constants.WGS84_CRS).geometry
    raster = GeoRaster2.from_wms("tests/data/google.xml", vector, resolution=1)
    assert raster.resolution() == 1
    assert raster.crs == constants.WEB_MERCATOR_CRS
    assert raster.footprint().difference(vector).area < 0.9
    # this new assert doesn't pass
    # assert raster.pixel_crop((10, 10, 100, 100))


def test_georaster_wms_vrt_with_destination_file():
    vector = GeoFeature.from_record(record, crs=constants.WGS84_CRS).geometry
    with NamedTemporaryFile(suffix='.vrt') as f:
        raster = GeoRaster2.from_wms("tests/data/google.xml", vector, resolution=1, destination_file=f.name)
        assert raster._filename == f.name
        assert os.path.exists(f.name)
        assert raster.resolution() == 1
        assert raster.crs == constants.WEB_MERCATOR_CRS
        assert raster.footprint().difference(vector).area < 0.9


def test_boundless_vrt():
    with rasterio.open("tests/data/raster/overlap2.tif") as raster:
        doc = boundless_vrt_doc(raster).tostring()
    with open("tests/data/raster/overlap2.vrt", 'rb') as expected_src:
        expected = expected_src.read()
        assert expected == doc


def test_boundless_vrt_preserves_mask():
    expected = GeoRaster2.open("tests/data/raster/overlap1.tif")
    with rasterio.open(expected.source_file) as raster:
        doc = boundless_vrt_doc(raster, bands=[1, 2]).tostring()
        with TemporaryDirectory() as d:
            file_name = os.path.join(d, 'vrt_file.vrt')
            with open(file_name, 'wb') as f:
                f.write(doc)
            raster = GeoRaster2.open(file_name)
            assert np.array_equal(raster.image.mask, expected.image.mask[:2, :, :])


def test_raster_list_vrt_for_single_raster():
    rasters = [
        GeoRaster2.open("tests/data/raster/overlap1.tif"),
    ]
    doc = raster_list_vrt(rasters, False).tostring()
    with TemporaryDirectory() as d:
        file_name = os.path.join(d, 'vrt_file.vrt')
        with open(file_name, 'wb') as f:
            f.write(doc)
        raster = GeoRaster2.open(file_name)
        assert raster.crs == rasters[0].crs
        assert raster.shape == rasters[0].shape
        assert raster.affine.almost_equals(rasters[0].affine)
        assert (raster.image.data == rasters[0].image.data).all()
        assert raster.band_names == rasters[0].band_names


def test_vrt_from_single_raster():
    rasters = [
        GeoRaster2.open("tests/data/raster/overlap1.tif"),
    ]
    raster = GeoRaster2.from_rasters(rasters, False)
    assert raster.crs == rasters[0].crs
    assert raster.shape == rasters[0].shape
    assert raster.affine.almost_equals(rasters[0].affine)
    assert (raster.image.data == rasters[0].image.data).all()


def test_vrt_from_multi_raster():
    rasters = [
        GeoRaster2.open("tests/data/raster/overlap1.tif"),
        GeoRaster2.open("tests/data/raster/overlap2.tif")
    ]
    raster = GeoRaster2.from_rasters(rasters, False)
    # made by gdalbuildvrt
    expected = GeoRaster2.open("tests/data/raster/expected_overlaps.vrt")
    assert raster.crs == expected.crs
    assert raster.shape == expected.shape
    # is this reasonable
    relative_precission = 10e-6 * expected.resolution()
    assert raster.affine.almost_equals(expected.affine, precision=relative_precission)
    assert (raster.image == expected.image).all()


def test_vrt_from_multi_raster_and_save_to_file():
    rasters = [
        GeoRaster2.open("tests/data/raster/overlap1.tif"),
        GeoRaster2.open("tests/data/raster/overlap2.tif")
    ]
    with NamedTemporaryFile(suffix='.vrt') as f:

        raster = GeoRaster2.from_rasters(rasters, False, destination_file=f.name)
        assert raster._filename == f.name
        assert os.path.exists(f.name)
        # made by gdalbuildvrt
        expected = GeoRaster2.open("tests/data/raster/expected_overlaps.vrt")
        assert raster.crs == expected.crs
        assert raster.shape == expected.shape
        # is this reasonable
        relative_precission = 10e-6 * expected.resolution()
        assert raster.affine.almost_equals(expected.affine, precision=relative_precission)
        assert (raster.image == expected.image).all()


def test_vrt_from_rasters_feature_collection():
    rasters = [
        GeoRaster2.open("tests/data/raster/overlap1.tif"),
        GeoRaster2.open("tests/data/raster/overlap2.tif")
    ]
    fc = FeatureCollection.from_georasters(rasters)
    raster = GeoRaster2.from_rasters(fc, False)
    # made by gdalbuildvrt
    expected = GeoRaster2.open("tests/data/raster/expected_overlaps.vrt")
    assert raster.crs == expected.crs
    assert raster.shape == expected.shape
    # is this reasonable
    relative_precission = 10e-6 * expected.resolution()
    assert raster.affine.almost_equals(expected.affine, precision=relative_precission)
    assert (raster.image == expected.image).all()


def test_build_vrt():
    source_file = 'tests/data/raster/rgb.tif'
    with NamedTemporaryFile(suffix='.vrt') as fp:
        vrt = build_vrt(source_file, fp.name)
        assert GeoRaster2.open(source_file) == GeoRaster2.open(vrt)
