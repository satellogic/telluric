import os
import copy
import os.path
import warnings
from collections.abc import Sequence
from collections import OrderedDict, defaultdict
from functools import partial
from itertools import islice, chain
from typing import Set, Iterator, Dict, Callable, Optional, Any, Union, DefaultDict

import fiona
from fiona.io import MemoryFile
from fiona.schema import FIELD_TYPES_MAP_REV
from shapely.geometry import CAP_STYLE
from rasterio.crs import CRS
from shapely.prepared import prep

from telluric.constants import WEB_MERCATOR_CRS, WGS84_CRS
from telluric.util.general import as_crs
from telluric.vectors import GeoVector
from telluric.features import GeoFeature
from telluric.plotting import NotebookPlottingMixin


DRIVERS = {
    '.json': 'GeoJSON',
    '.geojson': 'GeoJSON',
    '.shp': 'ESRI Shapefile'
}

MAX_WORKERS = int(os.environ.get('TELLURIC_LIB_MAX_WORKERS', 5))
CONCURRENCY_TIMEOUT = int(os.environ.get('TELLURIC_LIB_CONCURRENCY_TIMEOUT', 600))


def dissolve(collection, aggfunc=None):
    # type: (BaseCollection, Optional[Callable[[list], Any]]) -> GeoFeature
    """Dissolves features contained in a FeatureCollection and applies an aggregation
    function to its properties.

    """
    new_properties = {}
    if aggfunc:
        temp_properties = defaultdict(list)  # type: DefaultDict[Any, Any]
        for feature in collection:
            for key, value in feature.properties.items():
                temp_properties[key].append(value)

        for key, values in temp_properties.items():
            try:
                new_properties[key] = aggfunc(values)

            except Exception:
                # We just do not use these results
                pass

    return GeoFeature(collection.cascaded_union, new_properties)


class BaseCollection(Sequence, NotebookPlottingMixin):

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):  # type: () -> Iterator[GeoFeature]
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __eq__(self, other):
        return all(feat == feat_other for feat, feat_other in zip(self, other))

    def __add__(self, other):
        if isinstance(other, GeoVector):
            other = GeoFeature(other, {})

        if isinstance(other, GeoFeature):
            other = [other]

        return FeatureCollection([feat for feat in chain(self, other)])

    @property
    def crs(self):
        raise NotImplementedError

    @property
    def property_names(self):
        return list(self.schema['properties'].keys()) if self else []

    @property
    def schema(self):
        raise NotImplementedError

    @property
    def is_empty(self):
        """True if all features are empty."""
        return all(feature.is_empty for feature in self)

    @property
    def cascaded_union(self):  # type: () -> GeoVector
        return GeoVector.cascaded_union(self.geometries, self.crs, True)

    @property
    def convex_hull(self):  # type: () -> GeoVector
        return self.cascaded_union.convex_hull

    @property
    def envelope(self):  # type: () -> GeoVector
        # This is not exactly equal as cascaded_union,
        # as we are computing the envelope of the envelopes,
        # hence saving time
        return GeoVector.envelopes_union(self.geometries, self.crs, True)

    @property
    def geometries(self):
        return [feature.geometry for feature in self]

    @property
    def __geo_interface__(self):
        return self.to_record(WGS84_CRS)

    def to_record(self, crs):
        return {
            'type': 'FeatureCollection',
            'features': [feature.to_record(crs) for feature in self],
        }

    def get_values(self, key):
        """Get all values of a certain property.

        """
        for feature in self:
            yield feature.get(key)

    def reproject(self, new_crs):
        return FeatureCollection([feature.reproject(new_crs) for feature in self])

    def filter(self, intersects):
        """Filter results that intersect a given GeoFeature or Vector.

        """
        try:
            crs = self.crs
            vector = intersects.geometry if isinstance(intersects, GeoFeature) else intersects
            prepared_shape = prep(vector.get_shape(crs))
            hits = []

            for feature in self:
                target_shape = feature.geometry.get_shape(crs)
                if prepared_shape.overlaps(target_shape) or prepared_shape.intersects(target_shape):
                    hits.append(feature)

        except IndexError:
            hits = []

        return FeatureCollection(hits)

    def sort(self, by, desc=False):
        """Sorts by given property or function, ascending or descending order.

        Parameters
        ----------
        by : str or callable
            If string, property by which to sort.
            If callable, it should receive a GeoFeature a return a value by which to sort.
        desc : bool, optional
            Descending sort, default to False (ascending).

        """
        if callable(by):
            key = by
        else:
            def key(feature):
                return feature[by]

        sorted_features = sorted(list(self), reverse=desc, key=key)
        return FeatureCollection(sorted_features)

    def groupby(self, by):
        # type: (Union[str, Callable[[GeoFeature], str]]) -> _CollectionGroupBy
        """Groups collection using a value of a property.

        Parameters
        ----------
        by : str or callable
            If string, name of the property by which to group.
            If callable, should receive a GeoFeature and return the category.

        Returns
        -------
        _CollectionGroupBy

        """
        results = OrderedDict()  # type: OrderedDict[str, list]
        for feature in self:
            if callable(by):
                value = by(feature)
            else:
                value = feature[by]

            results.setdefault(value, []).append(feature)

        if hasattr(self, "_schema"):
            # I am doing this to trick mypy, is there a better way?
            # calling self._schema generates a mypy problem
            schema = getattr(self, "_schema")

        return _CollectionGroupBy(results, schema=schema)

    def dissolve(self, by=None, aggfunc=None):
        # type: (Optional[str], Optional[Callable]) -> FeatureCollection
        """Dissolve geometries and rasters within `groupby`.

        """
        if by:
            agg = partial(dissolve, aggfunc=aggfunc)  # type: Callable[[BaseCollection], GeoFeature]
            return self.groupby(by).agg(agg)

        else:
            return FeatureCollection([dissolve(self, aggfunc)])

    def map(self, map_function):
        """Return a new FeatureCollection with the results of applying `map_function` to each element.

        """
        return FeatureCollection(map_function(x) for x in self)

    def rasterize(self, dest_resolution, *, polygonize_width=0, crs=WEB_MERCATOR_CRS, fill_value=None,
                  bounds=None, dtype=None, **polygonize_kwargs):
        """Binarize a FeatureCollection and produce a raster with the target resolution.

        Parameters
        ----------
        dest_resolution: float
            Resolution in units of the CRS.
        polygonize_width : int, optional
            Width for the polygonized features (lines and points) in pixels, default to 0 (they won't appear).
        crs : ~rasterio.crs.CRS, dict (optional)
            Coordinate system, default to :py:data:`telluric.constants.WEB_MERCATOR_CRS`.
        fill_value : float or function, optional
            Value that represents data, default to None (will default to :py:data:`telluric.rasterization.FILL_VALUE`.
            If given a function, it must accept a single :py:class:`~telluric.features.GeoFeature` and return a numeric
            value.
        nodata_value : float, optional
            Nodata value, default to None (will default to :py:data:`telluric.rasterization.NODATA_VALUE`.
        bounds : GeoVector, optional
            Optional bounds for the target image, default to None (will use the FeatureCollection convex hull).
        dtype : numpy.dtype, optional
            dtype of the result, required only if fill_value is a function.
        polygonize_kwargs : dict
            Extra parameters to the polygonize function.

        """
        # Avoid circular imports
        from telluric.georaster import merge_all, MergeStrategy
        from telluric.rasterization import rasterize, NODATA_DEPRECATION_WARNING

        # Compute the size in real units and polygonize the features
        if not isinstance(polygonize_width, int):
            raise TypeError("The width in pixels must be an integer")

        if polygonize_kwargs.pop("nodata_value", None):
            warnings.warn(NODATA_DEPRECATION_WARNING, DeprecationWarning)

        # If the pixels width is 1, render points as squares to avoid missing data
        if polygonize_width == 1:
            polygonize_kwargs.update(cap_style_point=CAP_STYLE.square)

        # Reproject collection to target CRS
        if (
            self.crs is not None and
            self.crs != crs
        ):
            reprojected = self.reproject(crs)
        else:
            reprojected = self

        width = polygonize_width * dest_resolution
        polygonized = [feature.polygonize(width, **polygonize_kwargs) for feature in reprojected]

        # Discard the empty features
        shapes = [feature.geometry.get_shape(crs) for feature in polygonized
                  if not feature.is_empty]

        if bounds is None:
            bounds = self.envelope

        if bounds.area == 0.0:
            raise ValueError("Specify non-empty ROI")

        if not len(self):
            fill_value = None

        if callable(fill_value):
            if dtype is None:
                raise ValueError("dtype must be specified for multivalue rasterization")

            rasters = []
            for feature in self:
                rasters.append(feature.geometry.rasterize(
                    dest_resolution, fill_value=fill_value(feature), bounds=bounds, dtype=dtype, crs=crs)
                )

            return merge_all(rasters, bounds.reproject(crs), dest_resolution, merge_strategy=MergeStrategy.INTERSECTION)

        else:
            return rasterize(shapes, crs, bounds.get_shape(crs), dest_resolution, fill_value=fill_value, dtype=dtype)

    def _adapt_feature_before_write(self, feature):
        return feature

    def save(self, filename, driver=None, schema=None):
        """Saves collection to file.

        """
        if driver is None:
            driver = DRIVERS.get(os.path.splitext(filename)[-1])

        if schema is None:
            schema = self.schema

        if driver == "GeoJSON":
            crs = WGS84_CRS
        else:
            crs = self.crs

        # https://github.com/rasterio/rasterio/issues/2453
        # https://github.com/rasterio/rasterio/issues/3282 (to_dict() is lossy in rasterio)
        # We should switch to fiona 1.9.x and convert to fiona.crs.CRS
        crs_wkt = crs.to_wkt()

        with fiona.open(filename, 'w', driver=driver, schema=schema, crs_wkt=crs_wkt) as sink:
            for feature in self:
                new_feature = self._adapt_feature_before_write(feature)
                sink.write(new_feature.to_record(crs))

    def is_rasters_collection(self):
        if self.is_empty:
            return False
        return any(f.has_raster for f in self)

    def apply(self, **kwargs):
        """Return a new FeatureCollection with the results of applying the statements in the arguments to each element.

        """
        def _apply(f):
            properties = copy.deepcopy(f.properties)
            for prop, value in kwargs.items():
                if callable(value):
                    properties[prop] = value(f)
                else:
                    properties[prop] = value

            return f.copy_with(properties=properties)
        new_fc = self.map(_apply)
        new_schema = self.schema.copy()
        property_names_set = kwargs.keys()
        prop_types_map = FeatureCollection.guess_types_by_feature(new_fc[0], property_names_set)
        for key, value_type in prop_types_map.items():
            # already defined attribute that we just override will have the same position as before
            # new attributes will be appened
            new_schema["properties"][key] = FIELD_TYPES_MAP_REV.get(value_type, 'str')
        new_fc._schema = new_schema
        return new_fc


class FeatureCollectionIOError(BaseException):
    pass


class FeatureCollection(BaseCollection):

    def __init__(self, results, schema=None):
        """Initialize FeatureCollection object.

        Parameters
        ----------
        results : iterable
            Iterable of :py:class:`~telluric.features.GeoFeature` objects.

        """
        super().__init__()
        self._results = list(results)
        self._schema = schema
        self.validate()

    def validate(self):
        """
        if schema exists we run shape file validation code of fiona by trying to save to in MemoryFile
        """
        if self._schema is not None:
            with MemoryFile() as memfile:
                with memfile.open(driver="ESRI Shapefile", schema=self.schema) as target:
                    for _item in self._results:
                        # getting rid of the assets that don't behave well becasue of in memroy rasters
                        item = GeoFeature(_item.geometry, _item.properties)
                        target.write(item.to_record(item.crs))

    def __len__(self):
        return len(self._results)

    def __iter__(self):
        yield from self._results

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(self._results[index])
        return self._results[index]

    def __repr__(self):
        return str(list(self))

    @property
    def crs(self):
        # Get the CRS from the first feature
        return self._results[0].crs if self._results else None

    @staticmethod
    def guess_types_by_feature(feature, property_names_set):
        prop_types_map = dict([
            (prop_name, type(feature.get(prop_name)))
            for prop_name in property_names_set])
        return prop_types_map

    def _compute_properties(self):
        property_names_set = set()  # type: Set[str]
        for feat in self:
            property_names_set = property_names_set.union(feat.properties)

        # make type mapping based on the first feature
        prop_types_map = FeatureCollection.guess_types_by_feature(self[0], property_names_set)

        for feat in self:
            for prop_name, prop_value in feat.items():
                if prop_value is not None:
                    if isinstance(None, prop_types_map[prop_name]):
                        prop_types_map[prop_name] = type(prop_value)
                    if not isinstance(prop_value, prop_types_map[prop_name]):
                        raise FeatureCollectionIOError(
                            "Cannot generate a schema for a heterogeneous FeatureCollection. "
                            "Please convert all the appropriate properties to the same type."
                        )

        properties = OrderedDict(
            (k, FIELD_TYPES_MAP_REV.get(v, 'str'))
            for k, v in prop_types_map.items()
        )

        return properties

    @property
    def schema(self):
        if self._schema is None:
            if not self.is_empty:
                all_geometries = {feat.geometry.type for feat in self}
                if len(all_geometries) > 1:
                    raise FeatureCollectionIOError(
                        "Cannot generate a schema for a heterogeneous FeatureCollection. "
                        "Please convert all the geometries to the same type."
                    )
                else:
                    geometry = all_geometries.pop()

                self._schema = {
                    'geometry': geometry,
                    'properties': self._compute_properties()
                }
            else:
                self._schema = {"geometry": None, "properties": {}}

        return self._schema

    @classmethod
    def from_geovectors(cls, geovectors):
        """Builds new FeatureCollection from a sequence of :py:class:`~telluric.vectors.GeoVector` objects."""
        return cls([GeoFeature(vector, {}) for vector in geovectors])

    @classmethod
    def from_georasters(cls, georasters):
        """Builds new FeatureCollection from a sequence of :py:class:`~telluric.georaster.GeoRaster2` objects."""
        return cls([GeoFeature.from_raster(raster, {}) for raster in georasters])

    @classmethod
    def from_record(cls, record, crs):
        features = record.get("features", [])
        features = [GeoFeature.from_record(f, crs) for f in features]
        return cls(features)

    def _adapt_feature_before_write(self, feature):
        new_properties = feature.properties.copy()
        for key in self.property_names:
            new_properties.setdefault(key, None)

        new_geometry = feature.geometry.reproject(self.crs)

        return GeoFeature(new_geometry, new_properties)


class FileCollection(BaseCollection):
    """FileCollection object.

    """

    def __init__(self, filename, crs, schema, length):
        """Initialize a FileCollection object.

        Use the :py:meth:`~telluric.collections.FileCollection.open()` method instead.

        """
        super().__init__()
        self._fp = filename
        self._crs = crs
        self._schema = schema
        self._length = length

    def __eq__(self, other):
        return (
            all(feat == feat_other for feat, feat_other in zip(self, other))
            and len(self) == len(other)
            and self.crs == other.crs
            and self.property_names == self.property_names
        )

    @classmethod
    def open(cls, filename, crs=None):
        """Creates a FileCollection from a file in disk.

        Parameters
        ----------
        filename : str
            Path of the file to read.
        crs : CRS
            overrides the crs of the collection, this funtion will not reprojects

        """
        with fiona.Env():
            with fiona.open(filename, 'r') as source:
                original_crs = as_crs(source.crs)
                schema = source.schema
                length = len(source)
        crs = crs or original_crs
        ret_val = cls(filename, crs, schema, length)
        return ret_val

    @property
    def crs(self):
        return self._crs

    @property
    def schema(self):
        return self._schema

    def __len__(self):
        return self._length

    def __iter__(self):
        if hasattr(self._fp, 'seek'):
            self._fp.seek(0)
        with fiona.open(self._fp, 'r') as source:
            for record in source:
                yield GeoFeature.from_record(record, self.crs, source.schema)

    def __getitem__(self, index):
        # See https://github.com/Toblerity/Fiona/issues/327 for discussion
        # about random access in Fiona
        def adjust(bound):
            if bound is not None:
                if bound > 0 and bound > len(self):
                    bound = len(self)
                elif bound < 0 and abs(bound) >= len(self):
                    bound = 0
            return bound

        if isinstance(index, int):
            if (
                (index < 0 and abs(index) > len(self)) or
                (index >= 0 and index >= len(self))
            ):
                raise IndexError("collection index out of range")

            # We have to convert to positive indices to use islice here, see
            # https://bugs.python.org/issue33040
            index = index % len(self)
            return list(islice(self, index, index + 1))[0]

        else:
            start, stop, step = adjust(index.start), adjust(index.stop), index.step

            if start is not None and start < 0:
                start = start % len(self)
            if stop is not None and stop < 0:
                stop = stop % len(self)

            try:
                results = list(islice(self, start, stop, step))

            except ValueError:
                # Some value is negative, open the whole file first and then slice
                # Optimizing this requires some non trivial logic, which essentially means
                # reimplementing the whole Python backwards indexing code
                results = list(self)[index]

            return FeatureCollection(results)


class _CollectionGroupBy:

    def __init__(self, groups, schema=None):
        # type: (Dict, Optional[Dict]) -> None
        self._groups = groups
        self._schema = schema

    def __getitem__(self, key):
        results = OrderedDict.fromkeys(self._groups)
        for name, group in self:
            results[name] = []
            for feature in group:
                new_feature = GeoFeature(
                    feature.geometry,
                    {key: feature[key]}
                )
                results[name].append(new_feature)  # type: ignore

        return self.__class__(results)

    def __iter__(self):
        for name, group in self._groups.items():
            yield name, FeatureCollection(group, schema=self._schema)

    def agg(self, func):
        # type: (Callable[[BaseCollection], GeoFeature]) -> FeatureCollection
        """Apply some aggregation function to each of the groups.

        The function must take a FeatureCollection and produce a Feature.

        """
        return FeatureCollection(func(fc) for _, fc in self)

    def filter(self, func):
        # type: (Callable[[BaseCollection], bool]) -> _CollectionGroupBy
        """Filter out Groups based on filtering function.

        The function should get a FeatureCollection and return True to leave in the Group and False to take it out.
        """
        results = OrderedDict()  # type: OrderedDict
        for name, group in self:
            if func(group):
                results[name] = group

        return self.__class__(results)
