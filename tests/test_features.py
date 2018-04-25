import pytest
from datetime import datetime
import json

from shapely.geometry import Point, Polygon, LineString, mapping

from telluric.vectors import (
    GeoVector,
    GEOM_PROPERTIES, GEOM_UNARY_PREDICATES, GEOM_BINARY_PREDICATES, GEOM_BINARY_OPERATIONS
)

from telluric.features import GeoFeature


def test_geofeature_initializer():
    expected_geovector = GeoVector(Point(0.0, 0.0))
    expected_attributes = {'attribute_1': 1}

    res = GeoFeature(expected_geovector, expected_attributes)

    assert res.geometry is expected_geovector
    assert dict(res) == expected_attributes


def test_geofeature_str():
    expected_geovector = GeoVector(Point(0.0, 0.0))
    expected_attributes = {'attribute_1': 1}

    res = GeoFeature(expected_geovector, expected_attributes)

    assert str(res) == "GeoFeature(Point, {'attribute_1': 1})"


def test_geofeature_dict():
    expected_geovector = GeoVector(Point(0.0, 0.0))
    expected_attributes = {'attribute_1': 1}

    res = GeoFeature(expected_geovector, expected_attributes)

    assert res['attribute_1'] == 1
    assert res.geometry == expected_geovector


def test_geofeature_geo_interface():
    shape = Point(0.0, 10.0)
    crs = {'init': 'epsg:32630'}
    attributes = {
        'attribute_1': '1',
        'attribute_2': 'a',
    }
    gv = GeoVector(shape, crs)

    expected_geo_interface = {
        'type': 'Feature',
        'geometry': gv.__geo_interface__,
        'properties': attributes
    }

    feature = GeoFeature(gv, attributes)

    geo_interface = mapping(feature)

    assert geo_interface.keys() == expected_geo_interface.keys()
    assert geo_interface['type'] == expected_geo_interface['type']
    assert geo_interface['properties'] == expected_geo_interface['properties']


@pytest.mark.parametrize("property_name", GEOM_PROPERTIES)
def test_delegated_properties(property_name):
    expected_attributes = {'attribute_1': 1}
    feature = GeoFeature(
        GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
        expected_attributes
    )

    assert getattr(feature, property_name).geometry == getattr(feature.geometry, property_name)
    assert getattr(feature, property_name).attributes == feature.attributes


@pytest.mark.parametrize("property_name", ['x', 'y', 'xy'])
def test_delegated_properties(property_name):
    feature = GeoFeature(
        GeoVector(Point(0, 10)),
        {}
    )

    assert getattr(feature, property_name) == getattr(feature.geometry, property_name)


@pytest.mark.parametrize("predicate_name", GEOM_UNARY_PREDICATES)
def test_delegated_unary_predicates(predicate_name):
    feature = GeoFeature(
        GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
        {}
    )

    assert getattr(feature, predicate_name) == getattr(feature.geometry, predicate_name)


@pytest.mark.parametrize("predicate_name", GEOM_BINARY_PREDICATES)
def test_delegated_predicates(predicate_name):
    feature_1 = GeoFeature.from_shape(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    feature_2 = GeoFeature.from_shape(Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]))

    assert (getattr(feature_1, predicate_name)(feature_2) ==
            getattr(feature_1.geometry, predicate_name)(feature_2.geometry))


@pytest.mark.parametrize("operation_name", GEOM_BINARY_OPERATIONS)
def test_delegated_operations(operation_name):
    attributes_1 = {'attribute_1': 1}
    attributes_2 = {'attribute_2': 2}
    feature_1 = GeoFeature(
        GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
        attributes_1
    )
    feature_2 = GeoFeature(
        GeoVector(Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])),
        attributes_2
    )
    expected_attributes = {'attribute_1': 1, 'attribute_2': 2}

    assert (getattr(feature_1, operation_name)(feature_2).geometry ==
            getattr(feature_1.geometry, operation_name)(feature_2.geometry))

    assert getattr(feature_1, operation_name)(feature_2).attributes == expected_attributes
    assert getattr(feature_2, operation_name)(feature_1).attributes == expected_attributes


@pytest.mark.parametrize("operation_name", GEOM_BINARY_OPERATIONS)
def test_geofeature_and_geovector(operation_name):
    expected_attributes = {'attribute_1': 1}
    feature = GeoFeature(
        GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
        expected_attributes
    )
    vector = GeoVector(Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]))

    assert (getattr(feature, operation_name)(vector).geometry ==
            getattr(feature.geometry, operation_name)(vector))

    assert getattr(feature, operation_name)(vector).attributes == expected_attributes


def test_polygonize_is_equivalent_to_geovector():
    line = LineString([(1, 1), (0, 0)])
    feature = GeoFeature.from_shape(line)

    assert feature.polygonize(1).geometry == feature.geometry.polygonize(1)
    assert feature.polygonize(1).attributes == feature.attributes


@pytest.mark.parametrize("geometry", [
    GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
    GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 2)])),
])
@pytest.mark.parametrize("attributes", [
    {'attribute_1': 1},
    {'attribute_1': 2}
])
def test_geofeature_equality_checks_geometry_and_attributes(geometry, attributes):
    ref_geometry = GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    ref_attributes = {'attribute_1': 1}

    feature1 = GeoFeature(
        ref_geometry,
        ref_attributes
    )
    feature2 = GeoFeature(
        geometry,
        attributes
    )

    if geometry == ref_geometry and attributes == ref_attributes:
        assert feature1 == feature2

    else:
        assert feature1 != feature2


def test_geofeature_correctly_serializes_non_simple_types():
    feature = GeoFeature(
        GeoVector(Point(0, 0)),
        {'attr1': 1, 'attr2': '2', 'attr3': datetime(2018, 4, 25, 11, 18)}
    )
    expected_properties = {
        'attr1': 1, 'attr2': '2', 'attr3': '2018-04-25 11:18:00'
    }
    expected_json = ('{"type": "Feature", "properties": {"attr1": 1, "attr2": "2", "attr3": "2018-04-25 11:18:00"}, '
                     '"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}')

    assert mapping(feature)['properties'] == expected_properties
    assert json.dumps(mapping(feature)) == expected_json
