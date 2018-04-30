import os
from collections import OrderedDict
from datetime import date, datetime

import fiona
import tempfile
import mercantile

import pytest
from unittest import mock

from shapely.geometry import Polygon, Point, mapping

from telluric.constants import DEFAULT_CRS, WGS84_CRS, WEB_MERCATOR_CRS
from telluric.vectors import GeoVector
from telluric.features import GeoFeature
from telluric.collections import FeatureCollection, FileCollection, FeatureCollectionIOError
from telluric.georaster import GeoRaster2, merge_all


def fc_generator(num_features):
    gen_features = (
        GeoFeature.from_shape(
            Polygon([(0 + d_x, 0), (0 + d_x, 1), (1 + d_x, 1), (1 + d_x, 0)])
        )
        for d_x in range(num_features)
    )
    return FeatureCollection(gen_features)


def test_feature_collection_geo_interface():
    num_features = 3
    gen_features = (GeoFeature.from_shape(Point(0.0, 0.0))
                    for _ in range(num_features))
    fcol = FeatureCollection(gen_features)

    expected_geo_interface = {
        'type': 'FeatureCollection',
        'features': [
            feature.__geo_interface__ for feature in fcol
        ]
    }

    geo_interface = mapping(fcol)

    assert geo_interface.keys() == expected_geo_interface.keys()
    assert geo_interface['type'] == expected_geo_interface['type']
    assert geo_interface['features'] == expected_geo_interface['features']


def test_feature_collection_filter_returns_proper_elements():
    num_features = 3
    gen_features = (
        GeoFeature.from_shape(
            Polygon([(0 + d_x, 0), (0 + d_x, 1), (1 + d_x, 1), (1 + d_x, 0)])
        )
        for d_x in range(num_features)
    )

    filter_gv = GeoVector(Point(1.5, 0.5))

    fcol = FeatureCollection(gen_features)
    res = fcol.filter(intersects=filter_gv)

    assert len(list(res)) == 1


def test_convex_hull_and_envelope():
    fc = FeatureCollection.from_geovectors([
        GeoVector.from_bounds(xmin=0, ymin=0, xmax=1, ymax=1),
        GeoVector.from_bounds(xmin=1, ymin=0, xmax=2, ymax=1),
        GeoVector.from_bounds(xmin=1, ymin=1, xmax=2, ymax=2),
    ])
    expected_convex_hull = GeoVector(Polygon([(0, 0), (2, 0), (2, 2), (1, 2), (0, 1), (0, 0)]))
    expected_envelope = GeoVector.from_bounds(xmin=0, ymin=0, xmax=2, ymax=2)

    assert fc.convex_hull.equals(expected_convex_hull)
    assert fc.envelope.equals(expected_envelope)


def test_convex_hull_raises_warning_with_invalid_shape():
    # Invalid coordinates
    coords = [(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)]
    shape = Polygon(coords)
    gv = GeoFeature.from_shape(shape)
    assert not shape.is_valid

    fcol = FeatureCollection([gv])

    with pytest.warns(UserWarning) as record:
        fcol.convex_hull

    assert len(record) == 1
    assert record[0].message.args[0] == "Some invalid shapes found, discarding them."


def test_featurecollection_attribute_names_includes_all():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(0, 0)), {'attr1': 1}),
        GeoFeature(GeoVector(Point(0, 0)), {'attr2': 1})
    ])

    expected_attribute_names = ['attr1', 'attr2']

    assert sorted(fc.attribute_names) == expected_attribute_names


def test_featurecollection_schema_raises_error_for_heterogeneous_types():
    fc = FeatureCollection.from_geovectors([
        GeoVector(Polygon.from_bounds(0, 0, 1, 1)),
        GeoVector(Point(0, 0))
    ])

    with pytest.raises(FeatureCollectionIOError) as excinfo:
        fc.schema

    assert "Cannot generate a schema for a heterogeneous FeatureCollection. " in excinfo.exconly()


def test_featurecollection_map():
    fc = fc_generator(5)

    def func(feat):
        return GeoFeature(
            feat.geometry,
            {'attr1': 1}
        )

    new_fc = fc.map(func)

    assert new_fc == FeatureCollection([func(feat) for feat in fc])


@pytest.mark.parametrize("fc", [
    fc_generator(num_features=5),
    FileCollection.open("tests/data/vector/bsas_barrios_lla.geojson"),
])
def test_collection_slicing(fc):
    features = list(fc)

    # Single item
    assert fc[0] == features[0]
    assert fc[-1] == features[-1]

    # Positive step slicing
    assert fc[1:3] == FeatureCollection(features[1:3])
    assert fc[:2] == FeatureCollection(features[:2])
    assert fc[-2:] == FeatureCollection(features[-2:])
    assert fc[:-1] == FeatureCollection(features[:-1])
    assert fc[::2] == FeatureCollection(features[::2])

    # Negative step slicing
    assert fc[::-1] == FeatureCollection(features[::-1])
    assert fc[::-2] == FeatureCollection(features[::-2])
    assert fc[5::-2] == FeatureCollection(features[5::-2])
    assert fc[:3:-1] == FeatureCollection(features[:3:-1])
    assert fc[5:1:-2] == FeatureCollection(features[5:1:-2])


@mock.patch('telluric.collections.rasterize')
def test_rasterize_without_bounds(mock_rasterize):
    fc = fc_generator(num_features=1)
    fc.rasterize(dest_resolution=0.1, crs=DEFAULT_CRS, fill_value=29, nodata_value=-19)
    f = next(iter(fc))
    expected_shape = [f.geometry.get_shape(f.geometry.crs)]
    expected_bounds = f.geometry.get_shape(f.geometry.crs)
    mock_rasterize.assert_called_with(expected_shape, DEFAULT_CRS,
                                      expected_bounds, 0.1,
                                      29, -19)


@mock.patch('telluric.collections.rasterize')
def test_rasterize_with_geovector_bounds(mock_rasterize):
    fc = fc_generator(num_features=1)
    expected_bounds = Polygon.from_bounds(0, 0, 1, 1)
    bounds = GeoVector(expected_bounds, crs=DEFAULT_CRS)
    fc.rasterize(0.00001, crs=DEFAULT_CRS, bounds=bounds)
    f = next(iter(fc))
    expected_shape = [f.geometry.get_shape(f.geometry.crs)]
    mock_rasterize.assert_called_with(expected_shape, DEFAULT_CRS,
                                      expected_bounds, 0.00001,
                                      None, None)


@mock.patch('telluric.collections.rasterize')
def test_rasterize_with_polygon_bounds(mock_rasterize):
    fc = fc_generator(num_features=1)
    expected_bounds = Polygon.from_bounds(0, 0, 1, 1)
    bounds = expected_bounds
    fc.rasterize(0.00001, crs=DEFAULT_CRS, bounds=bounds)
    f = next(iter(fc))
    expected_shape = [f.geometry.get_shape(f.geometry.crs)]
    mock_rasterize.assert_called_with(expected_shape, DEFAULT_CRS,
                                      expected_bounds, 0.00001,
                                      None, None)


def test_file_collection_open():
    expected_len = 53
    expected_attribute_names = ['BARRIO', 'COMUNA', 'PERIMETRO', 'AREA']

    fcol = FileCollection.open("tests/data/vector/bsas_barrios_lla.geojson")

    assert len(fcol) == expected_len
    assert fcol.attribute_names == expected_attribute_names
    assert fcol.crs == WGS84_CRS


def test_file_save_geojson_twice():
    fcol = FileCollection.open("tests/data/vector/bsas_barrios_lla.geojson")

    with tempfile.NamedTemporaryFile(suffix=".json") as fp:
        fcol.save(fp.name)
        fcol.save(fp.name)


def test_file_collection_open_save_geojson():
    fcol = FileCollection.open("tests/data/vector/bsas_barrios_lla.geojson")

    with tempfile.NamedTemporaryFile(suffix=".json") as fp:
        fcol.save(fp.name)
        fcol_res = FileCollection.open(fp.name)

        assert fcol_res.crs == WGS84_CRS
        assert fcol == fcol_res


def test_file_collection_open_save_respects_projection():
    fcol = fc_generator(4).reproject(WEB_MERCATOR_CRS)

    with tempfile.NamedTemporaryFile(suffix=".json") as fp:
        fcol.save(fp.name)
        fcol_res = FileCollection.open(fp.name)

        assert fcol_res.crs == WGS84_CRS
        assert fcol == fcol_res.reproject(fcol.crs)


@pytest.mark.xfail(reason="Schema generation is not yet fully compliant")
def test_file_collection_open_save_shapefile():
    fcol = FileCollection.open("tests/data/vector/creaf/42111/42111.shp")[:10]

    with tempfile.NamedTemporaryFile(suffix=".json") as fp:
        fcol.save(fp.name)
        fcol_res = FileCollection.open(fp.name)

        import time
        time.sleep(5)
        assert fcol_res.crs == WGS84_CRS
        assert fcol == fcol_res.reproject(fcol.crs)


def test_feature_collection_with_dates_serializes_correctly():
    # "For Shapefiles, however, the only possible field type is 'date' as 'datetime' and 'time' are not available."
    # https://github.com/Toblerity/Fiona/pull/130
    # See also: https://github.com/Toblerity/Fiona/issues/572
    schema = {
        'geometry': 'Point',
        'properties': OrderedDict([
            ('prop_date', 'date'),
        ]),
    }
    expected_attributes = {
        'prop_date': date(2018, 4, 23),
    }
    feature = GeoFeature(GeoVector(Point(0, 0)), expected_attributes)
    with tempfile.TemporaryDirectory() as path:
        file_path = os.path.join(path, "test_dates.shp")
        with fiona.open(file_path, mode='w', driver="ESRI Shapefile", schema=schema, crs=feature.crs) as sink:
            sink.write(mapping(feature))

        fc = FileCollection.open(file_path)

        assert fc.schema == schema
        assert fc[0].geometry == feature.geometry
        assert fc[0].attributes == expected_attributes


@pytest.mark.parametrize("tile", [(4377, 3039, 13), (4376, 3039, 13), (4377, 3039, 13),
                                  (2189, 1519, 12), (8756, 6076, 14), (8751, 6075, 14)])
def test_get_tile_merge_tiles(tile):
    tile = (4377, 3039, 13)
    raster1_path = './tests/data/raster/overlap1.tif'
    raster2_path = './tests/data/raster/overlap2.tif'
    raster1 = GeoRaster2.open(raster1_path)
    raster2 = GeoRaster2.open(raster2_path)

    features = [
        GeoFeature(raster1.footprint(), {'raster_url': raster1_path, 'created': datetime.now()}),
        GeoFeature(raster2.footprint(), {'raster_url': raster2_path, 'created': datetime.now()}),
    ]
    fc = FeatureCollection(features)
    bounds = mercantile.xy_bounds(*tile)
    eroi = GeoVector.from_bounds(xmin=bounds.left, xmax=bounds.right,
                                 ymin=bounds.bottom, ymax=bounds.top,
                                 crs=WEB_MERCATOR_CRS)
    expected_tile = merge_all([raster1.get_tile(*tile), raster2.get_tile(*tile)], roi=eroi)
    merged = fc.get_tile(*tile, sort_by='created')
    assert merged == expected_tile
