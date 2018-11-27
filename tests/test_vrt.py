from telluric import GeoFeature, GeoRaster2, constants
from rasterio.io import MemoryFile
from telluric.vrt import wms_vrt, prettify
import os

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
    assert raster._filename.startswith("/vsimem")
    assert raster._filename.endswith(".vrt")
