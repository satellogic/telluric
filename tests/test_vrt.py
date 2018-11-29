import os
import rasterio
from telluric import GeoFeature, GeoRaster2, constants
from rasterio.io import MemoryFile
from telluric.vrt import wms_vrt, boundless_vrt_doc
from telluric.base_vrt import prettify


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


def test_wms_vrt():
    vector = GeoFeature.from_record(record, crs=constants.WGS84_CRS).geometry
    doc = str(wms_vrt("tests/data/google.xml",
                      bounds=vector.get_bounds(constants.WEB_MERCATOR_CRS),
                      resolution=1))
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


def test_boundless_vrt():
    with rasterio.open("tests/data/raster/overlap2.tif") as raster:
        doc = boundless_vrt_doc(raster)
        # with open("tests/data/raster/overlap2.vrt", 'wb') as dst:
        #     dst.write(doc)
    with open("tests/data/raster/overlap2.vrt", 'rb') as expected_src:
        expected = expected_src.read()
        assert expected == doc
