import os
import os.path
import warnings
import contextlib
from collections import Sequence
from itertools import islice
from typing import Set, Iterator

import fiona
from shapely.geometry import CAP_STYLE
from rasterio.crs import CRS
from shapely.ops import cascaded_union
from shapely.prepared import prep

import mercantile
from concurrent.futures import ThreadPoolExecutor

from telluric.constants import DEFAULT_CRS, WEB_MERCATOR_CRS, WGS84_CRS
from telluric.plotting import NotebookPlottingMixin
from telluric.rasterization import rasterize
from telluric.vectors import GeoVector
from telluric.features import GeoFeature
from telluric.georaster import merge_all, mercator_zoom_to_resolution

DRIVERS = {
    '.json': 'GeoJSON',
    '.geojson': 'GeoJSON',
    '.shp': 'ESRI Shapefile'
}

MAX_WORKERS = int(os.environ.get('TELLURIC_LIB_MAX_WORKERS', 5))
CONCURRENCY_TIMEOUT = int(os.environ.get('TELLURIC_LIB_CONCURRENCY_TIMEOUT', 600))


class BaseCollection(Sequence, NotebookPlottingMixin):
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):  # type: () -> Iterator[GeoFeature]
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __eq__(self, other):
        return all(feat == feat_other for feat, feat_other in zip(self, other))

    @property
    def crs(self):
        raise NotImplementedError

    @property
    def attribute_names(self):
        return list(self.schema['properties'].keys())

    @property
    def schema(self):
        raise NotImplementedError

    @property
    def is_empty(self):
        """True if all features are empty."""
        return all(feature.is_empty for feature in self)

    @property
    def convex_hull(self):  # type: () -> GeoVector
        try:
            crs = self.crs
            shapes = [feature.geometry.get_shape(crs) for feature in self]

            if not all([sh.is_valid for sh in shapes]):
                warnings.warn(
                    "Some invalid shapes found, discarding them."
                )

        except IndexError:
            crs = DEFAULT_CRS
            shapes = []

        return GeoVector(
            cascaded_union([sh for sh in shapes if sh.is_valid]).convex_hull,
            crs=crs
        )

    @property
    def envelope(self):  # type: () -> GeoVector
        try:
            crs = self.crs
            envelopes = [feature.geometry.envelope.get_shape(crs) for feature in self]

            if not all([sh.is_valid for sh in envelopes]):
                warnings.warn(
                    "Some invalid shapes found, discarding them."
                )

        except IndexError:
            crs = DEFAULT_CRS
            envelopes = []

        return GeoVector(
            cascaded_union([sh for sh in envelopes if sh.is_valid]).envelope,
            crs=crs
        )

    @property
    def __geo_interface__(self):
        return self.to_record(WGS84_CRS)

    def to_record(self, crs):
        return {
            'type': 'FeatureCollection',
            'features': [feature.to_record(crs) for feature in self],
        }

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

    def map(self, map_function):
        """Return a new FeatureCollection with the results of applying `map_function` to each element.

        """
        return FeatureCollection(map_function(x) for x in self)

    def rasterize(self, dest_resolution, polygonize_width=0, crs=WEB_MERCATOR_CRS, fill_value=None,
                  nodata_value=None, bounds=None, **polygonize_kwargs):
        """Binarize a FeatureCollection and produce a raster with the target resolution.

        Parameters
        ----------
        dest_resolution: float
            Resolution in units of the CRS.
        polygonize_width : int, optional
            Width for the polygonized features (lines and points) in pixels, default to 0 (they won't appear).
        crs : ~rasterio.crs.CRS, dict (optional)
            Coordinate system, default to :py:data:`telluric.constants.WEB_MERCATOR_CRS`.
        fill_value : float, optional
            Value that represents data, default to None (will default to :py:data:`telluric.rasterization.FILL_VALUE`.
        nodata_value : float, optional
            Nodata value, default to None (will default to :py:data:`telluric.rasterization.NODATA_VALUE`.
        bounds : GeoVector, optional
            Optional bounds for the target image, default to None (will use the FeatureCollection convex hull).
        polygonize_kwargs : dict
            Extra parameters to the polygonize function.

        """
        # Compute the size in real units and polygonize the features
        if not isinstance(polygonize_width, int):
            raise TypeError("The width in pixels must be an integer")

        # If the pixels width is 1, render points as squares to avoid missing data
        if polygonize_width == 1:
            polygonize_kwargs.update(cap_style_point=CAP_STYLE.square)

        width = polygonize_width * dest_resolution
        polygonized = [feature.polygonize(width, **polygonize_kwargs) for feature in self]

        # Discard the empty features
        shapes = [feature.geometry.get_shape(crs) for feature in polygonized
                  if not feature.is_empty]

        if bounds is None:
            bounds = self.convex_hull.get_shape(crs)

        if bounds.area == 0.0:
            raise ValueError("Specify non-empty ROI")

        elif isinstance(bounds, GeoVector):
            bounds = bounds.get_shape(crs)

        return rasterize(shapes, crs, bounds, dest_resolution, fill_value, nodata_value)

    def plot(self, mp=None, max_plot_rows=200, **plot_kwargs):
        if len(self) > max_plot_rows:
            warnings.warn(
                "Plotting only first {num_rows} rows to avoid browser freeze, "
                "please use .filter to narrow your query."
                .format(num_rows=max_plot_rows)
            )
            subset = self[:max_plot_rows]
            return subset.plot(mp, **plot_kwargs)
        else:
            return super().plot(mp, **plot_kwargs)

    def _adapt_feature_before_write(self, feature):
        return feature

    def save(self, filename, driver=None):
        """Saves collection to file.

        """
        if driver is None:
            driver = DRIVERS.get(os.path.splitext(filename)[-1])

        if driver == "GeoJSON":
            # Workaround for https://github.com/Toblerity/Fiona/issues/438
            # https://stackoverflow.com/a/27045091/554319
            with contextlib.suppress(FileNotFoundError):
                os.remove(filename)

            crs = WGS84_CRS
        else:
            crs = self.crs

        with fiona.open(filename, 'w', driver=driver, schema=self.schema, crs=crs) as sink:
            for feature in self:
                new_feature = self._adapt_feature_before_write(feature)
                sink.write(new_feature.to_record(crs))

    executer = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def get_tile(self, x, y, z, sort_by=None, desc=False, bands=None):
        """Generate mercator tile from rasters in FeatureCollection.

        Parameters
        ----------
        x: int
            x coordinate of tile
        y: int
            y coordinate of tile
        z: int
            zoom level
        sort_by: str
            attribute in feature to sort by
        desc: bool
            True for descending order, False for ascending
        bands: list
            list of indices of requested bads, default None which returns all bands

        Returns
        -------
        GeoRaster2

        """
        bb = mercantile.xy_bounds(x, y, z)
        roi = GeoVector.from_bounds(xmin=bb.left, ymin=bb.bottom,
                                    xmax=bb.right, ymax=bb.top,
                                    crs=WEB_MERCATOR_CRS)

        filtered_fc = self.filter(roi)

        def _get_tiled_feature(feature):
            return feature.get_tiled_feature(x, y, z, bands)

        tiled_features = list(self.executer.map(_get_tiled_feature,
                                                filtered_fc,
                                                timeout=CONCURRENCY_TIMEOUT))

        # tiled_features can be sort for different merge strategies
        if sort_by is not None:
            tiled_features = sorted(tiled_features,
                                    reverse=desc,
                                    key=lambda f: f[sort_by])

        tiles = [f['tile'] for f in tiled_features]
        if tiles:
            resolution = mercator_zoom_to_resolution[z]
            tile = merge_all(tiles, roi, dest_resolution=resolution)
            return tile
        else:
            return None


class FeatureCollectionIOError(BaseException):
    pass


class FeatureCollection(BaseCollection):
    def __init__(self, results):
        """Initialize FeatureCollection object.

        Parameters
        ----------
        results : list
            List of :py:class:`~telluric.features.GeoFeature` objects.

        """
        super().__init__()
        self._results = list(results)
        self._schema = None

    def __len__(self):
        return len(self._results)

    def __iter__(self):
        yield from self._results

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(self._results[index])
        return self._results[index]

    @property
    def crs(self):
        # Get the CRS from the first feature
        return self._results[0].crs

    def _compute_attributes(self):
        attribute_names_set = set()  # type: Set[str]
        for feat in self:
            attribute_names_set = attribute_names_set.union(feat.attributes)

        # make type mapping based on the first feature
        attr_types_map = dict([
            (attr_name, type(self[0].get(attr_name)))
            for attr_name in attribute_names_set])

        for feat in self:
            for attr_name, attr_value in feat.items():
                if attr_value is not None:
                    if isinstance(None, attr_types_map[attr_name]):
                        attr_types_map[attr_name] = type(attr_value)
                    if not isinstance(attr_value, attr_types_map[attr_name]):
                        raise FeatureCollectionIOError(
                            "Cannot generate a schema for a heterogeneous FeatureCollection. "
                            "Please convert all the appropriate properties to the same type."
                        )

        fiona_field_types_map = dict([
            (v, k) for k, v in fiona.FIELD_TYPES_MAP.items()])
        properties = {
            k: fiona_field_types_map.get(v) or 'str'
            for k, v in attr_types_map.items()
        }

        return properties

    @property
    def schema(self):
        if self._schema is None:
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
                'properties': self._compute_attributes()
            }

        return self._schema

    @classmethod
    def from_geovectors(cls, geovectors):
        """Builds new FeatureCollection from a sequence of :py:class:`~telluric.vectors.GeoVector` objects."""
        return cls([GeoFeature(vector, {}) for vector in geovectors])

    def _adapt_feature_before_write(self, feature):
        new_attributes = feature.attributes.copy()
        for key in self.attribute_names:
            new_attributes.setdefault(key, None)

        new_geometry = feature.geometry.reproject(self.crs)

        return GeoFeature(new_geometry, new_attributes)


class FileCollection(BaseCollection):
    """FileCollection object.

    """
    def __init__(self, filename, crs, schema, length):
        """Initialize a FileCollection object.

        Use the :py:meth:`~telluric.collections.FileCollection.open()` method instead.

        """
        super().__init__()
        self._filename = filename
        self._crs = crs
        self._schema = schema
        self._length = length

    def __eq__(self, other):
        return (
            all(feat == feat_other for feat, feat_other in zip(self, other))
            and len(self) == len(other)
            and self.crs == other.crs
            and self.attribute_names == self.attribute_names
        )

    @classmethod
    def open(cls, filename):
        """Creates a FileCollection from a file in disk.

        Parameters
        ----------
        filename : str
            Path of the file to read.

        """
        with fiona.open(filename, 'r') as source:
            crs = CRS(source.crs)
            schema = source.schema
            length = len(source)

        return cls(filename, crs, schema, length)

    @property
    def crs(self):
        return self._crs

    @property
    def schema(self):
        return self._schema

    def __len__(self):
        return self._length

    def __iter__(self):
        with fiona.open(self._filename, 'r') as source:
            for record in source:
                yield GeoFeature.from_record(record, source.crs, source.schema)

    def __getitem__(self, index):
        # See https://github.com/Toblerity/Fiona/issues/327 for discussion
        # about random access in Fiona
        if isinstance(index, int):
            # We have to convert to positive indices to use islice here, see
            # https://bugs.python.org/issue33040
            index = index % len(self)
            return list(islice(self, index, index + 1))[0]

        else:
            start, stop, step = index.start, index.stop, index.step

            start = start % len(self) if start is not None else None
            stop = stop % len(self) if stop is not None else None

            try:
                results = list(islice(self, start, stop, step))

            except ValueError:
                # Some value is negative, open the whole file first and then slice
                # Optimizing this requires some non trivial logic, which essentially means
                # reimplementing the whole Python backwards indexing code
                results = list(self)[index]

            return FeatureCollection(results)
