from collections import OrderedDict

import pytest
from datetime import time, date, datetime
import json

from shapely.geometry import Point, Polygon, LineString, mapping

from telluric.vectors import (
    GeoVector,
    GEOM_PROPERTIES, GEOM_UNARY_PREDICATES, GEOM_BINARY_PREDICATES, GEOM_BINARY_OPERATIONS
)

from telluric.features import transform_properties, GeoFeature
from telluric.georaster import GeoRaster2, GeoMultiRaster, GeoRaster2Error
from telluric.constants import WGS84_CRS, RASTER_TYPE


def test_geofeature_initializer():
    expected_geovector = GeoVector(Point(0.0, 0.0))
    expected_properties = {'property_1': 1}

    res = GeoFeature(expected_geovector, expected_properties)

    assert res.geometry is expected_geovector
    assert dict(res) == expected_properties


def test_geofeature_str():
    expected_geovector = GeoVector(Point(0.0, 0.0))
    expected_properties = {'property_1': 1}

    res = GeoFeature(expected_geovector, expected_properties)

    assert str(res) == "GeoFeature(Point, {'property_1': 1})"


def test_geofeature_dict():
    expected_geovector = GeoVector(Point(0.0, 0.0))
    expected_properties = {'property_1': 1}

    res = GeoFeature(expected_geovector, expected_properties)

    assert res['property_1'] == 1
    assert res.geometry == expected_geovector


def test_geofeature_geo_interface():
    shape = Point(0.0, 10.0)
    crs = {'init': 'epsg:32630'}
    properties = {
        'property_1': '1',
        'property_2': 'a',
    }
    gv = GeoVector(shape, crs)

    expected_geo_interface = {
        'type': 'Feature',
        'geometry': gv.__geo_interface__,
        'properties': properties,
        'assets': {}
    }

    feature = GeoFeature(gv, properties)

    geo_interface = mapping(feature)

    assert geo_interface.keys() == expected_geo_interface.keys()
    assert geo_interface['type'] == expected_geo_interface['type']
    assert geo_interface['properties'] == expected_geo_interface['properties']


@pytest.mark.parametrize("property_name", GEOM_PROPERTIES)
def test_delegated_properties(property_name):
    expected_properties = {'property_1': 1}
    feature = GeoFeature(
        GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
        expected_properties
    )

    assert getattr(feature, property_name).geometry == getattr(feature.geometry, property_name)


@pytest.mark.parametrize("property_name", ['x', 'y', 'xy'])
def test_another_delegated_properties(property_name):
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
    properties_1 = {'property_1': 1}
    properties_2 = {'property_2': 2}
    feature_1 = GeoFeature(
        GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
        properties_1
    )
    feature_2 = GeoFeature(
        GeoVector(Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])),
        properties_2
    )
    expected_properties = {'property_1': 1, 'property_2': 2}

    assert (getattr(feature_1, operation_name)(feature_2).geometry ==
            getattr(feature_1.geometry, operation_name)(feature_2.geometry))

    assert getattr(feature_1, operation_name)(feature_2).properties == expected_properties
    assert getattr(feature_2, operation_name)(feature_1).properties == expected_properties


@pytest.mark.parametrize("operation_name", GEOM_BINARY_OPERATIONS)
def test_geofeature_and_geovector(operation_name):
    expected_properties = {'property_1': 1}
    feature = GeoFeature(
        GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
        expected_properties
    )
    vector = GeoVector(Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]))

    assert (getattr(feature, operation_name)(vector).geometry ==
            getattr(feature.geometry, operation_name)(vector))

    assert getattr(feature, operation_name)(vector).properties == expected_properties


def test_polygonize_is_equivalent_to_geovector():
    line = LineString([(1, 1), (0, 0)])
    feature = GeoFeature.from_shape(line)

    assert feature.polygonize(1).geometry == feature.geometry.polygonize(1)
    assert feature.polygonize(1).properties == feature.properties


@pytest.mark.parametrize("geometry", [
    GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
    GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 2)])),
])
@pytest.mark.parametrize("properties", [
    {'property_1': 1},
    {'property_1': 2}
])
def test_geofeature_equality_checks_geometry_and_properties(geometry, properties):
    ref_geometry = GeoVector(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    ref_properties = {'property_1': 1}

    feature1 = GeoFeature(
        ref_geometry,
        ref_properties
    )
    feature2 = GeoFeature(
        geometry,
        properties
    )

    if geometry == ref_geometry and properties == ref_properties:
        assert feature1 == feature2

    else:
        assert feature1 != feature2


def test_geofeature_correctly_serializes_non_simple_types():
    date_feature = datetime(2018, 4, 25, 11, 18)
    feature = GeoFeature(
        GeoVector(Point(0, 0)),
        OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', date_feature)])
    )
    expected_properties = OrderedDict([
        ('prop1', 1), ('prop2', '2'), ('prop3', date_feature.isoformat())
    ])

    assert mapping(feature)['properties'] == expected_properties


@pytest.mark.parametrize("raster", [
    "test_raster_with_url", "test_raster_with_no_url"
])
def test_geofeature_from_raster_returns_a_valid_feature(raster, request):
    raster = request.getfixturevalue(raster)
    properties = OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', datetime(2018, 4, 25, 11, 18))])
    feature = GeoFeature.from_raster(raster, properties=properties)
    assert feature.properties == properties
    assert feature.raster() == raster
    assert feature.raster().footprint() == feature.geometry


@pytest.mark.parametrize("raster", [
    "test_raster_with_url", "test_raster_with_no_url"
])
def test_geofeature_from_raster_to_record_should_not_have___object(raster, request):
    raster = request.getfixturevalue(raster)
    properties = OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', datetime(2018, 4, 25, 11, 18))])
    feature = GeoFeature.from_raster(raster, properties=properties)
    feature_record = feature.to_record(feature.crs)
    for _, asset in feature_record['assets'].items():
        assert '__object' not in asset


@pytest.mark.parametrize("raster", [
    "test_raster_with_url"
])
def test_geofeature_from_raster_serializes_with_assets(raster, request):
    raster = request.getfixturevalue(raster)
    properties = OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', datetime(2018, 4, 25, 11, 18))])
    feature = GeoFeature.from_raster(raster, properties=properties)
    assert mapping(feature)["assets"] == {'0': {'href': raster._filename, 'bands': raster.band_names,
                                                'type': RASTER_TYPE, 'product': 'visual'}}


@pytest.mark.parametrize("raster", [
    "test_raster_with_url"
])
def test_geofeature_from_record_for_a_record_with_raster(raster, request):
    raster = request.getfixturevalue(raster)
    properties = OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', datetime(2018, 4, 25, 11, 18))])
    feature = GeoFeature.from_raster(raster, properties=properties)
    feature2 = GeoFeature.from_record(feature.to_record(feature.crs), feature.crs)
    assert feature.raster() == feature2.raster()
    assert feature.geometry == feature2.geometry
    assert feature2.to_record(feature2.crs)['properties'] == feature.to_record(feature.crs)['properties']
    assert feature2.crs == feature2.raster().crs


def test_geofeature_from_multi_rasters_returns_a_valid_feature(request):
    rasters = [request.getfixturevalue("test_raster_with_url"),
               request.getfixturevalue("test_raster_with_url")]
    raster = GeoMultiRaster(rasters)
    properties = OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', datetime(2018, 4, 25, 11, 18))])
    feature = GeoFeature.from_raster(raster, properties=properties)
    assert feature.properties == properties
    assert feature.raster() == raster
    assert feature.raster() == GeoRaster2.from_rasters(rasters)


def test_geofeature_from_multi_raster_serializes_with_assets(request):
    rasters = [request.getfixturevalue("test_raster_with_url"),
               request.getfixturevalue("test_raster_with_url")]
    raster = GeoMultiRaster(rasters)
    properties = OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', datetime(2018, 4, 25, 11, 18))])
    feature = GeoFeature.from_raster(raster, properties=properties)
    assert mapping(feature)["assets"] == {
        '0': {'href': rasters[0]._filename, 'bands': rasters[0].band_names,
              'type': RASTER_TYPE, 'product': 'visual'},
        '1': {'href': rasters[1]._filename, 'bands': rasters[1].band_names,
              'type': RASTER_TYPE, 'product': 'visual'}
    }


def test_geofeature_from_record_for_a_record_with_multi_raster(request):
    rasters = [request.getfixturevalue("test_raster_with_url"),
               request.getfixturevalue("test_raster_with_url")]
    raster = GeoMultiRaster(rasters)
    properties = OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', datetime(2018, 4, 25, 11, 18))])
    feature = GeoFeature.from_raster(raster, properties=properties)
    feature2 = GeoFeature.from_record(feature.to_record(feature.crs), feature.crs)
    assert feature.has_raster
    assert feature2.has_raster
    assert feature.raster() == feature2.raster()
    assert feature.geometry == feature2.geometry
    assert feature2.to_record(feature2.crs)['properties'] == feature.to_record(feature.crs)['properties']
    assert feature2.crs == feature.crs


def test_geofeature_from_record_with_empty_rasters(request):
    record = {
        'geometry': mapping(Point(0.0, 1.1)),
        'properties': {},
        'raster': []
    }
    feature = GeoFeature.from_record(record, WGS84_CRS)
    assert isinstance(feature, GeoFeature)


def test_transform_properties():
    schema = {
        'properties': OrderedDict([
            ('prop1', 'time'),
            ('prop2', 'date'),
            ('prop3', 'datetime'),
            ('prop4', 'datetime')
        ])
    }
    expected_properties = OrderedDict([
        ('prop1', time(15, 0, 0)),
        ('prop2', date(2018, 5, 19)),
        ('prop3', datetime(2018, 5, 19, 15, 0)),
        ('prop4', None)
    ])
    assert transform_properties(OrderedDict([
        ('prop1', '15:00:00'),
        ('prop2', '2018-05-19'),
        ('prop3', '2018-05-19T15:00:00'),
        ('prop4', None)
    ]), schema) == expected_properties


def test_geofeature_with_raster_copy_with(request):
    raster = request.getfixturevalue("test_raster_with_url")
    properties = OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', datetime(2018, 4, 25, 11, 18))])
    feature = GeoFeature.from_raster(raster, properties=properties)
    feature_copy = feature.copy_with()
    assert feature == feature_copy
    assert id(feature.raster()) != id(feature_copy.raster())
    assert id(feature.properties) != id(feature_copy.properties)


def test_geofeature_with_raster_copy_with_updates_properties(request):
    raster = request.getfixturevalue("test_raster_with_url")
    properties = OrderedDict([('prop1', 1), ('prop2', '2')])
    feature = GeoFeature.from_raster(raster, properties=properties)
    new_properties = OrderedDict([('prop2', 1), ('prop3', '2')])
    feature_copy = feature.copy_with(properties=new_properties)
    assert feature != feature_copy
    assert feature.raster() == feature_copy.raster()
    assert id(feature.raster()) != id(feature_copy.raster())

    assert feature_copy.properties == {'prop1': 1, 'prop2': 1, 'prop3': '2'}


def test_geofeature_copy_with(request):
    feature = GeoFeature(
        GeoVector(Point(0, 0)),
        OrderedDict([('prop1', 1), ('prop2', '2'), ('prop3', datetime(2018, 4, 25, 11, 18))])
    )

    feature_copy = feature.copy_with()
    assert feature == feature_copy
    assert id(feature.geometry) != id(feature_copy.geometry)
    assert id(feature.properties) != id(feature_copy.properties)


def test_geofeature_copy_with_updates_properties(request):
    feature = GeoFeature(
        GeoVector(Point(0, 0)),
        OrderedDict([('prop1', 1), ('prop2', '2')])
    )
    new_properties = OrderedDict([('prop2', 1), ('prop3', '2')])
    feature_copy = feature.copy_with(properties=new_properties)
    assert feature != feature_copy
    assert feature.geometry == feature_copy.geometry
    assert id(feature.geometry) != id(feature_copy.geometry)

    assert feature_copy.properties == {'prop1': 1, 'prop2': 1, 'prop3': '2'}
