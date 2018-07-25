import os
from collections import OrderedDict
from datetime import date
from functools import partial

import fiona
import tempfile

import pytest
from unittest import mock

from shapely.geometry import Polygon, Point, mapping

from telluric.constants import DEFAULT_CRS, WGS84_CRS, WEB_MERCATOR_CRS
from telluric.vectors import GeoVector
from telluric.features import GeoFeature
from telluric.collections import FeatureCollection, FileCollection, FeatureCollectionIOError, dissolve


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


def test_convex_hull_and_envelope_and_cascaded_union():
    fc = FeatureCollection.from_geovectors([
        GeoVector.from_bounds(xmin=0, ymin=0, xmax=1, ymax=1),
        GeoVector.from_bounds(xmin=1, ymin=0, xmax=2, ymax=1),
        GeoVector.from_bounds(xmin=1, ymin=1, xmax=2, ymax=2),
    ])

    expected_cascaded_union = GeoVector(Polygon([(0, 0), (2, 0), (2, 2), (1, 2), (1, 1), (0, 1), (0, 0)]))
    expected_convex_hull = GeoVector(Polygon([(0, 0), (2, 0), (2, 2), (1, 2), (0, 1), (0, 0)]))
    expected_envelope = GeoVector.from_bounds(xmin=0, ymin=0, xmax=2, ymax=2)

    assert fc.convex_hull.equals(expected_convex_hull)
    assert fc.envelope.equals(expected_envelope)
    assert fc.cascaded_union.equals(expected_cascaded_union)


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


def test_featurecollection_property_names_includes_all():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 1}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop2': 1})
    ])

    expected_property_names = ['prop1', 'prop2']

    assert sorted(fc.property_names) == expected_property_names


def test_featurecollection_schema_raises_error_for_heterogeneous_geometry_types():
    fc = FeatureCollection.from_geovectors([
        GeoVector(Polygon.from_bounds(0, 0, 1, 1)),
        GeoVector(Point(0, 0))
    ])

    with pytest.raises(FeatureCollectionIOError) as excinfo:
        fc.schema

    assert "Cannot generate a schema for a heterogeneous FeatureCollection. " in excinfo.exconly()


def test_featurecollection_schema_raises_error_for_heterogeneous_property_types():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 1}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 1.0})
    ])

    with pytest.raises(FeatureCollectionIOError) as excinfo:
        fc.schema

    assert "Cannot generate a schema for a heterogeneous FeatureCollection. " in excinfo.exconly()


def test_featurecollection_schema_for_property_types_with_none_values():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': None, 'prop2': 1.0, 'prop3': 'A'}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 2, 'prop2': None, 'prop3': 'B'}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 3, 'prop2': 3.0, 'prop3': None})
    ])

    expected_schema = {
        'geometry': 'Point',
        'properties': {
            'prop1': 'int',
            'prop2': 'float',
            'prop3': 'str'
        }
    }

    assert fc.schema == expected_schema


def test_featurecollection_schema_for_property_types_without_none_values():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 1, 'prop2': 1.0, 'prop3': 'A'}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 2, 'prop2': 2.0, 'prop3': 'B'})
    ])

    expected_schema = {
        'geometry': 'Point',
        'properties': {
            'prop1': 'int',
            'prop2': 'float',
            'prop3': 'str'
        }
    }

    assert fc.schema == expected_schema


def test_featurecollection_schema_treat_unsupported_property_types_as_str():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': bool(0), 'prop2': date(2018, 5, 19)}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': bool(1), 'prop2': date(2018, 5, 20)})
    ])

    expected_schema = {
        'geometry': 'Point',
        'properties': {
            'prop1': 'str',
            'prop2': 'str'
        }
    }

    assert fc.schema == expected_schema


def test_featurecollection_map():
    fc = fc_generator(5)

    def func(feat):
        return GeoFeature(
            feat.geometry,
            {'prop1': 1}
        )

    new_fc = fc.map(func)

    assert new_fc == FeatureCollection([func(feat) for feat in fc])


def test_featurecollection_save_has_no_side_effects():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 1}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop2': 1})
    ])

    with tempfile.NamedTemporaryFile(suffix=".json") as fp:
        fc.save(fp.name)

        assert fc[0].properties == {'prop1': 1}
        assert fc[1].properties == {'prop2': 1}


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

    # Slicing should return another FeatureCollection
    assert type(fc[:]) is FeatureCollection


@mock.patch('telluric.collections.rasterize')
def test_rasterize_without_bounds(mock_rasterize):
    fc = fc_generator(num_features=1)
    fc.rasterize(dest_resolution=0.1, crs=DEFAULT_CRS, fill_value=29)
    f = next(iter(fc))
    expected_shape = [f.geometry.get_shape(f.geometry.crs)]
    expected_bounds = fc.envelope.get_shape(fc.crs)
    mock_rasterize.assert_called_with(expected_shape, DEFAULT_CRS,
                                      expected_bounds, 0.1,
                                      fill_value=29, dtype=None)


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
                                      fill_value=None, dtype=None)


def test_file_collection_open():
    expected_len = 53
    expected_property_names = ['BARRIO', 'COMUNA', 'PERIMETRO', 'AREA']

    fcol = FileCollection.open("tests/data/vector/bsas_barrios_lla.geojson")

    assert len(fcol) == expected_len
    assert fcol.property_names == expected_property_names
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
    expected_properties = {
        'prop_date': date(2018, 4, 23),
    }
    feature = GeoFeature(GeoVector(Point(0, 0)), expected_properties)
    with tempfile.TemporaryDirectory() as path:
        file_path = os.path.join(path, "test_dates.shp")
        with fiona.open(file_path, mode='w', driver="ESRI Shapefile", schema=schema, crs=feature.crs) as sink:
            sink.write(mapping(feature))

        fc = FileCollection.open(file_path)

        assert fc.schema == schema
        assert fc[0].geometry == feature.geometry
        assert fc[0].properties == expected_properties


def test_get_values():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 1}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 2}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop1': 3}),
        GeoFeature(GeoVector(Point(0, 0)), {'prop2': 1}),
    ])

    assert list(fc.get_values('prop1')) == [1, 2, 3, None]


def test_sort():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(3, 3)), {'prop1': 3}),
        GeoFeature(GeoVector(Point(1, 1)), {'prop1': 1}),
        GeoFeature(GeoVector(Point(2, 2)), {'prop1': 2})
    ])

    expected_fc = FeatureCollection([
        GeoFeature(GeoVector(Point(1, 1)), {'prop1': 1}),
        GeoFeature(GeoVector(Point(2, 2)), {'prop1': 2}),
        GeoFeature(GeoVector(Point(3, 3)), {'prop1': 3})
    ])

    assert fc.sort("prop1") == expected_fc


def test_groupby_has_proper_groups():
    gfa1 = GeoFeature(GeoVector(Point(3, 3)), {'prop1': 'a'})
    gfa2 = GeoFeature(GeoVector(Point(1, 1)), {'prop1': 'a'})
    gfb1 = GeoFeature(GeoVector(Point(2, 2)), {'prop1': 'b'})
    fc = FeatureCollection([gfa1, gfa2, gfb1])

    expected_groups = [
        ('a', FeatureCollection([gfa1, gfa2])),
        ('b', FeatureCollection([gfb1]))
    ]

    assert list(fc.groupby('prop1')) == expected_groups


def test_groupby_can_extract_property():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(3, 3)), {'prop1': 'a', 'b': 1}),
        GeoFeature(GeoVector(Point(1, 1)), {'prop1': 'a', 'b': 2}),
        GeoFeature(GeoVector(Point(2, 2)), {'prop1': 'b', 'b': 3})
    ])

    expected_groups = [
        ('a', FeatureCollection([
            GeoFeature(GeoVector(Point(3, 3)), {'b': 1}),
            GeoFeature(GeoVector(Point(1, 1)), {'b': 2}),
        ])),
        ('b', FeatureCollection([
            GeoFeature(GeoVector(Point(2, 2)), {'b': 3})
        ]))
    ]

    assert list(fc.groupby('prop1')['b']) == expected_groups


def test_groupby_agg_returns_expected_result():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(3, 3)), {'prop1': 'a', 'b': 1}),
        GeoFeature(GeoVector(Point(1, 1)), {'prop1': 'a', 'b': 2}),
        GeoFeature(GeoVector(Point(2, 2)), {'prop1': 'b', 'b': 3})
    ])

    def first(collection):
        return collection[0]

    expected_result = FeatureCollection([
        GeoFeature(GeoVector(Point(3, 3)), {'b': 1}),
        GeoFeature(GeoVector(Point(2, 2)), {'b': 3})
    ])

    assert list(fc.groupby('prop1')['b'].agg(first)) == expected_result


def test_groupby_with_dissolve():
    fc = FeatureCollection([
        GeoFeature(GeoVector.from_bounds(xmin=0, ymin=0, xmax=2, ymax=1, crs=DEFAULT_CRS), {'prop1': 'a', 'b': 1}),
        GeoFeature(GeoVector.from_bounds(xmin=1, ymin=0, xmax=3, ymax=1, crs=DEFAULT_CRS), {'prop1': 'a', 'b': 2}),
        GeoFeature(GeoVector.from_bounds(xmin=0, ymin=0, xmax=2, ymax=1, crs=DEFAULT_CRS), {'prop1': 'b', 'b': 3}),
    ])

    expected_result = FeatureCollection([
        GeoFeature(GeoVector.from_bounds(xmin=0, ymin=0, xmax=3, ymax=1, crs=DEFAULT_CRS), {'b': 3}),
        GeoFeature(GeoVector.from_bounds(xmin=0, ymin=0, xmax=2, ymax=1, crs=DEFAULT_CRS), {'b': 3}),
    ])

    assert fc.dissolve('prop1', sum) == fc.groupby('prop1').agg(partial(dissolve, aggfunc=sum)) == expected_result


def test_filter_group_by():
    fc = FeatureCollection([
        GeoFeature(GeoVector(Point(3, 3)), {'prop1': 'a', 'b': 1}),
        GeoFeature(GeoVector(Point(1, 1)), {'prop1': 'a', 'b': 2}),
        GeoFeature(GeoVector(Point(3, 3)), {'prop1': 'b', 'b': 3}),
        GeoFeature(GeoVector(Point(1, 1)), {'prop1': 'b', 'b': 1}),
        GeoFeature(GeoVector(Point(2, 2)), {'prop1': 'b', 'b': 2}),
    ])

    expected_groups = [
        ('b', FeatureCollection([
            GeoFeature(GeoVector(Point(3, 3)), {'b': 3}),
            GeoFeature(GeoVector(Point(1, 1)), {'b': 1}),
            GeoFeature(GeoVector(Point(2, 2)), {'b': 2})
        ]))
    ]

    groups = fc.groupby('prop1')

    def filter_func(fc):
        return sorted([b for b in fc.get_values('b')]) == [1, 2, 3]

    filtered_group = groups.filter(filter_func)
    assert list(filtered_group['b']) == expected_groups
