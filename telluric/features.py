import copy
import warnings
from datetime import datetime
from collections.abc import Mapping

from dateutil.parser import parse as parse_date

from shapely.geometry import shape

from telluric.constants import DEFAULT_CRS, WGS84_CRS, RASTER_TYPE
from telluric.vectors import (
    GeoVector,
    GEOM_PROPERTIES, GEOM_NONVECTOR_PROPERTIES, GEOM_UNARY_PREDICATES, GEOM_BINARY_PREDICATES, GEOM_BINARY_OPERATIONS
)
from telluric import GeoRaster2
from telluric.plotting import NotebookPlottingMixin


raster_types = [RASTER_TYPE]


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
        if isinstance(attr_value, datetime):
            new_properties[attr_name] = attr_value.isoformat()
        elif not isinstance(attr_value, (dict, list, tuple, str, int, float, bool, type(None))):
            # Property is not JSON-serializable according to this table
            # https://docs.python.org/3.4/library/json.html#json.JSONEncoder
            # so we convert to string
            new_properties[attr_name] = str(attr_value)
    return new_properties


class GeoFeatureError(Exception):
    pass


class GeoFeature(Mapping, NotebookPlottingMixin):
    """GeoFeature object.

    """
    def __init__(self, geovector, properties, assets=None):
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
        self.assets = assets or {}

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
        assets = copy.deepcopy(self.assets)
        # remove the refference to the raster object in the asset entery
        for asset in assets.values():
            asset.pop('__object', None)

        ret_val = {
            'type': 'Feature',
            'properties': serialize_properties(self.properties),
            'geometry': self.geometry.to_record(crs),
            'assets': assets
        }
        return ret_val

    @classmethod
    def from_record(cls, record, crs, schema=None):
        """Create GeoFeature from a record."""
        properties = cls._to_properties(record, schema)
        vector = GeoVector(shape(record['geometry']), crs)
        if record.get('raster'):
            assets = {k: dict(type=RASTER_TYPE, product='visual', **v) for k, v in record.get('raster').items()}
        else:
            assets = record.get('assets', {})
        return cls(vector, properties, assets)

    @staticmethod
    def _to_properties(record, schema):
        if schema is not None:
            properties = transform_properties(record["properties"], schema)
        else:
            properties = record["properties"]
        return properties

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

    def __getattr__(self, item):
        if item in GEOM_PROPERTIES:
            def delegated_(self_):
                return getattr(self_.geometry, item)

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

    def copy_with(self, geometry=None, properties=None, assets=None):
        """Generate a new GeoFeature with different geometry or preperties."""
        def copy_assets_object(asset):
            obj = asset.get("__object")
            if hasattr("copy", obj):
                new_obj = obj.copy()
            if obj:
                asset["__object"] = new_obj

        geometry = geometry or self.geometry.copy()
        new_properties = copy.deepcopy(self.properties)
        if properties:
            new_properties.update(properties)
        if not assets:
            assets = copy.deepcopy(self.assets)
            map(copy_assets_object, assets.values())
        else:
            assets = {}
        return self.__class__(geometry, new_properties, assets)

    @classmethod
    def from_shape(cls, shape):
        return cls(GeoVector(shape, DEFAULT_CRS), {})

    @classmethod
    def from_raster(cls, raster, properties, product='visual'):
        """Initialize a GeoFeature object with a GeoRaster

        Parameters
        ----------
        raster : GeoRaster
            the raster in the feature
        properties : dict
            Properties.
        product : str
            product associated to the raster
        """
        footprint = raster.footprint()
        assets = raster.to_assets(product=product)
        return cls(footprint, properties, assets)

    @property
    def has_raster(self):
        """True if any of the assets  is type 'raster'."""
        return any(asset.get('type') == RASTER_TYPE for asset in self.assets.values())

    def raster(self, name=None, **creteria):
        """Generates a GeoRaster2 object based on the asset name(key) or a creteria(protety name and value)."""
        if name:
            asset = self.assets[name]
            if asset["type"] in raster_types:
                __object = asset.get('__object')
                if isinstance(__object, GeoRaster2):
                    return __object
                else:
                    return GeoRaster2.from_assets([asset])
            else:
                return None

        if creteria:
            key = next(iter(creteria))
            value = creteria[key]
        else:
            # default creteria is to return a visual raster
            key = 'product'
            value = 'visual'

        rasters = {k: asset for k, asset in self.assets.items() if asset['type'] in raster_types}
        raster_list = list(rasters.values())
        # if there is only a single raster in the assetes and it hase a GeoRaster object serve the object
        if len(raster_list) == 1 and isinstance(raster_list[0].get('__object'), GeoRaster2):
            return raster_list[0].get('__object')
        rasters = {k: r for k, r in rasters.items() if r[key] == value}
        raster = GeoRaster2.from_assets(rasters)
        return raster
