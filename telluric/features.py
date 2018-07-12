import warnings
from collections import Mapping

from dateutil.parser import parse as parse_date

from shapely.geometry import shape

from telluric.constants import DEFAULT_CRS, WGS84_CRS
from telluric.vectors import (
    GeoVector,
    GEOM_PROPERTIES, GEOM_NONVECTOR_PROPERTIES, GEOM_UNARY_PREDICATES, GEOM_BINARY_PREDICATES, GEOM_BINARY_OPERATIONS
)
from telluric.plotting import NotebookPlottingMixin


def transform_properties(properties, schema):
    """Transform properties types according to a schema.

    Parameters
    ----------
    properties : dict
        Properties to transform.
    schema : dict
        Fiona schema containing the types.

    """
    new_properties = properties.copy()
    for prop_value, (prop_name, prop_type) in zip(new_properties.values(), schema["properties"].items()):
        if prop_value is None:
            continue
        elif prop_type == "time":
            new_properties[prop_name] = parse_date(prop_value).time()
        elif prop_type == "date":
            new_properties[prop_name] = parse_date(prop_value).date()
        elif prop_type == "datetime":
            new_properties[prop_name] = parse_date(prop_value)

    return new_properties


def serialize_properties(properties):
    """Serialize properties.

    Parameters
    ----------
    properties : dict
        Properties to serialize.

    """
    new_properties = properties.copy()
    for attr_name, attr_value in new_properties.items():
        if not isinstance(attr_value, (dict, list, tuple, str, int, float, bool, type(None))):
            # Property is not JSON-serializable according to this table
            # https://docs.python.org/3.4/library/json.html#json.JSONEncoder
            # so we convert to string
            new_properties[attr_name] = str(attr_value)

    return new_properties


class GeoFeature(Mapping, NotebookPlottingMixin):
    """GeoFeature object.

    """
    def __init__(self, geovector, properties):
        """Initialize a GeoFeature object.

        Parameters
        ----------
        geovector : GeoVector
            Geometry.
        properties : dict
            Properties.

        """
        self.geometry = geovector  # type: GeoVector
        self._properties = properties

    @property
    def crs(self):
        return self.geometry.crs

    @property
    def properties(self):
        return self._properties

    @property
    def attributes(self):
        warnings.warn(
            "GeoFeature.attributes is deprecated and will be removed, please use GeoFeature.properties instead",
            DeprecationWarning
        )
        return self.properties

    @property
    def __geo_interface__(self):
        return self.to_record(WGS84_CRS)

    def to_record(self, crs):
        return {
            'type': 'Feature',
            'properties': serialize_properties(self.properties),
            'geometry': self.geometry.to_record(crs),
        }

    @classmethod
    def from_record(cls, record, crs, schema=None):
        if schema is not None:
            properties = transform_properties(record["properties"], schema)
        else:
            properties = record["properties"]

        return cls(
            GeoVector(
                shape(record['geometry']),
                crs
            ),
            properties
        )

    def __len__(self):
        return len(self.properties)

    def __getitem__(self, item):
        return self.properties[item]

    def __iter__(self):
        return iter(self.properties)

    def __eq__(self, other):
        return (
            self.geometry == other.geometry
            and self.properties == other.properties
        )

    @classmethod
    def from_shape(cls, shape):
        return cls(GeoVector(shape, DEFAULT_CRS), {})

    def __getattr__(self, item):
        if item in GEOM_PROPERTIES:
            def delegated_(self_):
                return self_.__class__(getattr(self_.geometry, item), self_.properties)

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
                # Transform to a GeoFeature without properties if necessary
                if isinstance(other, GeoVector):
                    other = self_.__class__(other, {})

                return getattr(self_.geometry, item)(
                    other.reproject(self_.geometry.crs).geometry)

            delegated_predicate.__doc__ = getattr(self.geometry._shape, item).__doc__

            # Bind the attribute
            setattr(self.__class__, item, delegated_predicate)

        elif item in GEOM_BINARY_OPERATIONS:
            def delegated_operation(self_, other):
                # Transform to a GeoFeature without properties if necessary
                if isinstance(other, GeoVector):
                    other = self_.__class__(other, {})

                properties = self_.properties.copy()
                properties.update(other.properties)
                return self_.__class__(
                    getattr(self_.geometry, item)(other.reproject(self_.geometry.crs).geometry),
                    properties
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

    def polygonize(self, width, **kwargs):
        return self.__class__(
            self.geometry.polygonize(width, **kwargs),
            self.properties
        )

    def reproject(self, new_crs):
        return self.__class__(self.geometry.reproject(new_crs), self.properties)

    def __str__(self):
        return "{}({}, {})".format(
            self.__class__.__name__,
            self.geometry._shape.__class__.__name__,
            dict(self))

    def __repr__(self):
        return str(self)
