import copy
from collections import Mapping

from dateutil.parser import parse as parse_date

from shapely.geometry import shape

from telluric.constants import DEFAULT_CRS, WGS84_CRS
from telluric.vectors import (
    GeoVector,
    GEOM_PROPERTIES, GEOM_NONVECTOR_PROPERTIES, GEOM_UNARY_PREDICATES, GEOM_BINARY_PREDICATES, GEOM_BINARY_OPERATIONS
)
from telluric.georaster import GeoRaster2
from telluric.plotting import NotebookPlottingMixin


def transform_attributes(attributes, schema):
    """Transform attributes types according to a schema.

    Parameters
    ----------
    attributes : dict
        Attributes to transform.
    schema : dict
        Fiona schema containing the types.

    """
    new_attributes = attributes.copy()
    for attr_value, (attr_name, attr_type) in zip(new_attributes.values(), schema["properties"].items()):
        if attr_value is None:
            continue
        elif attr_type == "time":
            new_attributes[attr_name] = parse_date(attr_value).time()
        elif attr_type == "date":
            new_attributes[attr_name] = parse_date(attr_value).date()
        elif attr_type == "datetime":
            new_attributes[attr_name] = parse_date(attr_value)

    return new_attributes


def serialize_attributes(attributes):
    """Serialize attributes.

    Parameters
    ----------
    attributes : dict
        Attributes to serialize.

    """
    new_attributes = attributes.copy()
    for attr_name, attr_value in new_attributes.items():
        if not isinstance(attr_value, (dict, list, tuple, str, int, float, bool, type(None))):
            # Attribute is not JSON-serializable according to this table
            # https://docs.python.org/3.4/library/json.html#json.JSONEncoder
            # so we convert to string
            new_attributes[attr_name] = str(attr_value)

    return new_attributes


class GeoFeature(Mapping, NotebookPlottingMixin):
    """GeoFeature object.

    """
    def __init__(self, geovector, attributes):
        """Initialize a GeoFeature object.

        Parameters
        ----------
        geovector : GeoVector
            Geometry.
        attributes : dict
            Properties.

        """
        self.geometry = geovector  # type: GeoVector
        self._attributes = attributes
        self._raster = None

    @property
    def crs(self):
        return self.geometry.crs

    @property
    def attributes(self):
        return self._attributes

    @property
    def __geo_interface__(self):
        return self.to_record(WGS84_CRS)

    def to_record(self, crs):
        return {
            'type': 'Feature',
            'properties': serialize_attributes(self._attributes),
            'geometry': self.geometry.to_record(crs),
        }

    @classmethod
    def from_record(cls, record, crs, schema=None):
        if schema is not None:
            attributes = transform_attributes(record["properties"], schema)
        else:
            attributes = record["properties"]

        return cls(
            GeoVector(
                shape(record['geometry']),
                crs
            ),
            attributes
        )

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, item):
        return self.attributes[item]

    def __iter__(self):
        return iter(self.attributes)

    def __eq__(self, other):
        return (
            self.geometry == other.geometry
            and self.attributes == other.attributes
        )

    @classmethod
    def from_shape(cls, shape):
        return cls(GeoVector(shape, DEFAULT_CRS), {})

    def __getattr__(self, item):
        if item in GEOM_PROPERTIES:
            def delegated_(self_):
                return self_.__class__(getattr(self_.geometry, item), self_.attributes)

            # Use class docstring to properly translate properties, see
            # https://stackoverflow.com/a/38118315/554319
            delegated_.__doc__ = getattr(self.geometry._shape.__class__, item).__doc__

            # Transform to a property
            delegated_property = property(delegated_)

            # Bind the property
            setattr(self.__class__, item, delegated_property)

        elif item in GEOM_NONVECTOR_PROPERTIES:
            def delegated_(self_):
                return getattr(self_.get_shape(self_.crs), item)

            # Use class docstring to properly translate properties, see
            # https://stackoverflow.com/a/38118315/554319
            delegated_.__doc__ = getattr(self.geometry._shape.__class__, item).__doc__

            # Transform to a property
            delegated_property = property(delegated_)

            # Bind the property
            setattr(self.__class__, item, delegated_property)

        elif item in GEOM_UNARY_PREDICATES:
            def delegated_(self_):
                return getattr(self_.geometry, item)

            # Use class docstring to properly translate properties, see
            # https://stackoverflow.com/a/38118315/554319
            delegated_.__doc__ = getattr(self.geometry._shape.__class__, item).__doc__

            # Transform to a property
            delegated_property = property(delegated_)

            # Bind the property
            setattr(self.__class__, item, delegated_property)

        elif item in GEOM_BINARY_PREDICATES:
            def delegated_predicate(self_, other):
                # Transform to a GeoFeature without attributes if necessary
                if isinstance(other, GeoVector):
                    other = self_.__class__(other, {})

                return getattr(self_.geometry, item)(
                    other.reproject(self_.geometry.crs).geometry)

            delegated_predicate.__doc__ = getattr(self.geometry._shape, item).__doc__

            # Bind the attribute
            setattr(self.__class__, item, delegated_predicate)

        elif item in GEOM_BINARY_OPERATIONS:
            def delegated_operation(self_, other):
                # Transform to a GeoFeature without attributes if necessary
                if isinstance(other, GeoVector):
                    other = self_.__class__(other, {})

                attributes = self_.attributes.copy()
                attributes.update(other.attributes)
                return self_.__class__(
                    getattr(self_.geometry, item)(other.reproject(self_.geometry.crs).geometry),
                    attributes
                )

            delegated_operation.__doc__ = getattr(self.geometry._shape, item).__doc__

            # Bind the attribute
            setattr(self.__class__, item, delegated_operation)

        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__, item))

        # Return the newly bound attribute
        return getattr(self, item)

    def get_shape(self, crs):
        """Gets the underlying Shapely shape in a specified CRS."""
        return self.geometry.get_shape(crs)

    def get_raster(self, field_name='raster_url'):
        if self._raster is None:
            self._raster = GeoRaster2.open(self[field_name])

        return self._raster

    def polygonize(self, width, **kwargs):
        return self.__class__(
            self.geometry.polygonize(width, **kwargs),
            self.attributes
        )

    def reproject(self, new_crs):
        return self.__class__(self.geometry.reproject(new_crs), self.attributes)

    def __str__(self):
        return "{}({}, {})".format(
            self.__class__.__name__,
            self.geometry._shape.__class__.__name__,
            dict(self))

    def __repr__(self):
        return str(self)
