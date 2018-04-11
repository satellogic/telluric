import json

import numpy as np

import shapely.geometry
from shapely.geometry import (
    shape as to_shape,
    Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString,
    CAP_STYLE,
    mapping)

from rasterio.crs import CRS

from telluric.constants import DEFAULT_CRS, EQUAL_AREA_CRS, WGS84_CRS
from telluric.plotting import NotebookPlottingMixin
from telluric.util.projections import transform


# From shapely.geometry.base.BaseGeometry
GEOM_PROPERTIES = [
    'boundary',
    'centroid',
    'convex_hull',
    'envelope',  # bounding_box
    'exterior',
]
GEOM_BINARY_OPERATIONS = [
    'difference',
    'intersection',
    'symmetric_difference',
    'union',
]

GEOM_UNARY_OPERATIONS = [
    'buffer',
    'simplify',
]

GEOM_UNARY_PREDICATES = [
    'has_z',
    'is_empty',
    'is_ring',
    'is_closed',
    'is_simple',
    'is_valid',
]
GEOM_BINARY_PREDICATES = [
    'relate',
    'covers',
    'contains',
    'crosses',
    'disjoint',
    'equals',
    'intersects',
    'overlaps',
    'touches',
    'within',
    # 'equals_exact',  # Requires extra parameter
    # 'almost_equals',  # Requires extra parameter
    # 'relate_pattern',
]
GEOM_NONVECTOR_PROPERTIES = [
    'xy',
    'x', 'y',
    'coords',
    'exterior',
    'interiors',
]


def get_dimension(geometry):
    """Gets the dimension of a Fiona-like geometry element."""
    coordinates = geometry["coordinates"]
    type_ = geometry["type"]
    if type_ in ('Point',):
        return len(coordinates)
    elif type_ in ('LineString', 'MultiPoint'):
        return len(coordinates[0])
    elif type_ in ('Polygon', 'MultiLineString'):
        return len(coordinates[0][0])
    elif type_ in ('MultiPolygon',):
        return len(coordinates[0][0][0])
    else:
        raise ValueError("Invalid type '{}'".format(type_))


def generate_tile_coordinates(roi, num_tiles):
    """Yields N x M rectangular tiles for a region of interest.

    Parameters
    ----------
    roi : GeoVector
        Region of interest
    num_tiles : tuple
        Tuple (horizontal_tiles, vertical_tiles)

    Yields
    ------
    ~telluric.vectors.GeoVector

    """
    bounds = roi.get_shape(roi.crs).bounds

    x_range = np.linspace(bounds[0], bounds[2], num_tiles[0] + 1)
    y_range = np.linspace(bounds[1], bounds[3], num_tiles[1] + 1)

    for y_start, y_end in zip(y_range[:-1], y_range[1:]):
        for x_start, x_end in zip(x_range[:-1], x_range[1:]):
            new_roi = GeoVector(
                Polygon.from_bounds(x_start, y_start, x_end, y_end),
                roi.crs
            )

            yield new_roi


def generate_tile_coordinates_from_pixels(roi, scale, size):
    """Yields N x M rectangular tiles for a region of interest.

    Parameters
    ----------
    roi : GeoVector
        Region of interest
    scale : float
        Scale factor (think of it as pixel resolution)
    size : tuple
        Pixel size in (width, height) to be multiplied by the scale factor

    Yields
    ------
    ~telluric.vectors.GeoVector

    """
    if not all(isinstance(coord, int) for coord in size):
        raise ValueError("Pixel size must be a tuple of integers")

    width = size[0] * scale
    height = size[1] * scale

    minx, miny, maxx, maxy = roi.get_shape(roi.crs).bounds

    num_w = np.ceil((maxx - minx) / width)
    num_h = np.ceil((maxy - miny) / height)

    new_roi = GeoVector.from_bounds(
        xmin=minx, ymin=miny,
        xmax=minx + num_w * width, ymax=miny + num_h * height,
        crs=roi.crs
    )

    yield from generate_tile_coordinates(new_roi, (num_w, num_h))


class _GeoVectorDelegator:
    def __getattr__(self, item):
        if item in GEOM_PROPERTIES:
            def delegated_(self_):
                return self_.__class__(getattr(self_.get_shape(self_.crs), item), self_.crs)

            # Use class docstring to properly translate properties, see
            # https://stackoverflow.com/a/38118315/554319
            delegated_.__doc__ = getattr(self._shape.__class__, item).__doc__

            # Transform to a property
            delegated_property = property(delegated_)

            # Bind the property
            setattr(self.__class__, item, delegated_property)

        elif item in GEOM_NONVECTOR_PROPERTIES:
            def delegated_(self_):
                return getattr(self_.get_shape(self_.crs), item)

            # Use class docstring to properly translate properties, see
            # https://stackoverflow.com/a/38118315/554319
            delegated_.__doc__ = getattr(self._shape.__class__, item).__doc__

            # Transform to a property
            delegated_property = property(delegated_)

            # Bind the property
            setattr(self.__class__, item, delegated_property)

        elif item in GEOM_UNARY_PREDICATES:
            def delegated_(self_):
                return getattr(self_.get_shape(self_.crs), item)

            # Use class docstring to properly translate properties, see
            # https://stackoverflow.com/a/38118315/554319
            delegated_.__doc__ = getattr(self._shape.__class__, item).__doc__

            # Transform to a property
            delegated_property = property(delegated_)

            # Bind the property
            setattr(self.__class__, item, delegated_property)

        elif item in GEOM_BINARY_PREDICATES:
            def delegated_predicate(self_, other):
                return getattr(self_.get_shape(self_.crs), item)(
                    other.get_shape(self_.crs))

            delegated_predicate.__doc__ = getattr(self._shape, item).__doc__

            # Bind the attribute
            setattr(self.__class__, item, delegated_predicate)

        elif item in GEOM_BINARY_OPERATIONS:
            def delegated_operation(self_, other):
                return self_.__class__(
                    getattr(self_.get_shape(self_.crs), item)(other.get_shape(self_.crs)),
                    self_.crs
                )

            delegated_operation.__doc__ = getattr(self._shape, item).__doc__

            # Bind the attribute
            setattr(self.__class__, item, delegated_operation)

        elif item in GEOM_UNARY_OPERATIONS:
            # We rename to make MyPy happy
            def delegated_operation_special(self_, *args, **kwargs):
                return self_.__class__(
                    getattr(self_.get_shape(self_.crs), item)(*args, **kwargs),
                    self_.crs
                )

            delegated_operation_special.__doc__ = getattr(self._shape, item).__doc__

            # Bind the attribute
            setattr(self.__class__, item, delegated_operation_special)

        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__, item))

        # Return the newly bound attribute
        return getattr(self, item)


class GeoVector(_GeoVectorDelegator, NotebookPlottingMixin):
    """Geometric element with an associated CRS.

    This class has also all the properties and methods of :py:class:`shapely.geometry.BaseGeometry`.

    """
    # noinspection PyInitNewSignature,PyMissingConstructor
    def __init__(self, shape, crs=DEFAULT_CRS):
        """Initialize GeoVector.

        Parameters
        ----------
        shape : shapely.geometry.BaseGeometry
            Geometry.
        crs : ~rasterio.crs.CRS, dict (optional)
            Coordinate Reference System, default to :py:data:`telluric.constants.DEFAULT_CRS`.

        """
        self._shape = shape  # type: shapely.geometry.base.BaseGeometry
        self._crs = crs

    @classmethod
    def from_geojson(cls, filename):
        """Load vector from geojson."""
        with open(filename) as fd:
            geometry = json.load(fd)

        if 'type' not in geometry:
            raise TypeError("%s is not a valid geojson." % (filename,))

        return cls(to_shape(geometry), WGS84_CRS)

    def to_geojson(self, filename):
        """Save vector as geojson."""
        with open(filename, 'w') as fd:
            json.dump(self.to_record(WGS84_CRS), fd)

    @classmethod
    def point(cls, x, y, crs=DEFAULT_CRS):
            return cls(Point((x, y)), crs)

    @classmethod
    def line(cls, points, crs=DEFAULT_CRS):
            return cls(LineString(points), crs)

    @classmethod
    def from_bounds(cls, *, xmin, ymin, xmax, ymax, crs=DEFAULT_CRS):
        """Creates GeoVector object from bounds.

        This function only accepts keyword arguments.

        Parameters
        ----------
        xmin, ymin, xmax, ymax : float
            Bounds of the GeoVector.
        crs : ~rasterio.crs.CRS, dict
            Projection, default to :py:data:`telluric.constants.DEFAULT_CRS`.

        Examples
        --------
        >>> from telluric import GeoVector
        >>> GeoVector.from_bounds(xmin=0, ymin=0, xmax=1, ymax=1)
        GeoVector(shape=POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)), crs=CRS({'init': 'epsg:4326'}))
        >>> GeoVector.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)
        GeoVector(shape=POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)), crs=CRS({'init': 'epsg:4326'}))

        """
        return cls(Polygon.from_bounds(xmin, ymin, xmax, ymax), crs)

    @property
    def __geo_interface__(self):
        return self.to_record(WGS84_CRS)

    @property
    def crs(self):
        return self._crs

    @property
    def area(self):
        return self.get_shape(EQUAL_AREA_CRS).area

    @property
    def type(self):
        return self._shape.type

    def to_record(self, crs):
        data = mapping(self.get_shape(crs))
        if data['type'] == 'LinearRing':
            data['type'] = 'Polygon'
            data['coordinates'] = (data['coordinates'],)
        return data

    def get_shape(self, crs):
        """Gets the underlying Shapely shape in a specified CRS.

        This method deliberately does not have a default crs=self.crs
        to force the user to specify it.

        """
        if crs == self.crs:
            return self._shape
        else:
            return self.reproject(crs)._shape

    def _repr_svg_(self):
        return self._shape._repr_svg_()

    def reproject(self, new_crs):
        if new_crs == self.crs:
            return self
        else:
            new_shape = transform(self._shape, self._crs, new_crs)
            return self.__class__(new_shape, new_crs)

    def rasterize(self, dest_resolution, fill_value=None, nodata_value=None, bounds=None):
        # Import here to avoid circular imports
        from telluric import rasterization  # noqa
        crs = self.crs
        shapes = [self.get_shape(crs)]
        if bounds is None:
            bounds = self.envelope.get_shape(crs)
        elif isinstance(bounds, GeoVector):
            bounds = bounds.get_shape(crs)

        return rasterization.rasterize(shapes, crs, bounds, dest_resolution, fill_value, nodata_value)

    def equals_exact(self, other, tolerance):
        """ invariant to crs. """
        # This method cannot be delegated because it has an extra parameter
        return self._shape.equals_exact(other.get_shape(self.crs), tolerance=tolerance)

    def almost_equals(self, other, decimal=6):
        """ invariant to crs. """
        # This method cannot be delegated because it has an extra parameter
        return self._shape.almost_equals(other.get_shape(self.crs), decimal=decimal)

    def polygonize(self, width, cap_style_line=CAP_STYLE.flat, cap_style_point=CAP_STYLE.round):
        """Turns line or point into a buffered polygon."""
        shape = self._shape
        if isinstance(shape, (LineString, MultiLineString)):
            return self.__class__(
                shape.buffer(width / 2, cap_style=cap_style_line),
                self.crs
            )
        elif isinstance(shape, (Point, MultiPoint)):
            return self.__class__(
                shape.buffer(width / 2, cap_style=cap_style_point),
                self.crs
            )
        else:
            return self

    def __eq__(self, other):
        """ invariant to crs and topology."""
        return self.equals(other)

    def __str__(self):
        return '{cls}(shape={shape}, crs={crs})'.format(
            cls=self.__class__.__name__, shape=self._shape, crs=self.crs)

    def __repr__(self):
        return str(self)
