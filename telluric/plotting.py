"""Code for interactive vector plots.

"""
import warnings
from copy import copy
from statistics import median_low

import mercantile
from shapely.geometry import mapping

try:
    import folium
    from ipyleaflet import (
        Map, GeoJSON,
        basemaps
    )
except ImportError:
    warnings.warn(
        "Visualization dependencies not available, plotting will not work",
        ImportWarning,
        stacklevel=2,
    )

from telluric.constants import WGS84_CRS
try:
    from telluric.util.local_tile_server import TileServer
except ImportError:
    warnings.warn(
        "Visualization dependencies not available, local tile server will not work",
        ImportWarning,
        stacklevel=2,
    )


SIMPLE_PLOT_MAX_ROWS = 200


def simple_plot(feature, *, mp=None, **map_kwargs):
    """Plots a GeoVector in a simple Folium map.

    For more complex and customizable plots using Jupyter widgets,
    use the plot function instead.

    Parameters
    ----------
    feature : telluric.vectors.GeoVector, telluric.features.GeoFeature, telluric.collections.BaseCollection
        Data to plot.

    """
    # This import is here to avoid cyclic references
    from telluric.collections import BaseCollection

    if mp is None:
        mp = folium.Map(tiles="Stamen Terrain", **map_kwargs)

    if feature.is_empty:
        warnings.warn("The geometry is empty.")

    else:
        if isinstance(feature, BaseCollection):
            feature = feature[:SIMPLE_PLOT_MAX_ROWS]

        folium.GeoJson(mapping(feature), name='geojson', overlay=True).add_to(mp)
        shape = feature.envelope.get_shape(WGS84_CRS)
        mp.fit_bounds([shape.bounds[:1:-1], shape.bounds[1::-1]])

    return mp


def zoom_level_from_geometry(geometry, splits=4):
    """Generate optimum zoom level for geometry.

    Notes
    -----
    The obvious solution would be

    >>> mercantile.bounding_tile(*geometry.get_shape(WGS84_CRS).bounds).z

    However, if the geometry is split between two or four tiles,
    the resulting zoom level might be too big.

    """
    # This import is here to avoid cyclic references
    from telluric.vectors import generate_tile_coordinates

    # We split the geometry and compute the zoom level for each chunk
    levels = []
    for chunk in generate_tile_coordinates(geometry, (splits, splits)):
        levels.append(mercantile.bounding_tile(*chunk.get_shape(WGS84_CRS).bounds).z)

    # We now return the median value using the median_low function, which
    # always picks the result from the list
    return median_low(levels)


def style_element(element, style_func=None):
    # This import is here to avoid cyclic references
    from telluric.features import GeoFeature

    if hasattr(element, "properties") and style_func is not None:
        new_properties = copy(element.properties)
        new_properties["style"] = style_func(mapping(element))
        return GeoFeature(element.geometry, new_properties)

    else:
        # If there are no properties or no function, there's nothing to style
        return element


def layer_from_element(element, style_function=None):
    """Return Leaflet layer from shape.

    Parameters
    ----------
    element : telluric.vectors.GeoVector, telluric.features.GeoFeature, telluric.collections.BaseCollection
        Data to plot.

    """
    # This import is here to avoid cyclic references
    from telluric.collections import BaseCollection

    if isinstance(element, BaseCollection):
        styled_element = element.map(lambda feat: style_element(feat, style_function))

    else:
        styled_element = style_element(element, style_function)

    return GeoJSON(data=mapping(styled_element), name='GeoJSON')


def plot(feature, mp=None, style_function=None, **map_kwargs):
    """Plots a GeoVector in an ipyleaflet map.

    Parameters
    ----------
    feature : telluric.vectors.GeoVector, telluric.features.GeoFeature, telluric.collections.BaseCollection
        Data to plot.
    mp : ipyleaflet.Map, optional
        Map in which to plot, default to None (creates a new one).
    style_function : func
        Function that returns an style dictionary for
    map_kwargs : kwargs, optional
        Extra parameters to send to ipyleaflet.Map.

    """
    map_kwargs.setdefault('basemap', basemaps.Stamen.Terrain)
    if feature.is_empty:
        warnings.warn("The geometry is empty.")
        mp = Map(**map_kwargs) if mp is None else mp

    else:
        if mp is None:
            center = feature.envelope.centroid.reproject(WGS84_CRS)
            zoom = zoom_level_from_geometry(feature.envelope)

            mp = Map(center=(center.y, center.x), zoom=zoom, **map_kwargs)

        mp.add_layer(layer_from_element(feature, style_function))

    return mp


class NotebookPlottingMixin:
    def _run_in_tileserver(self, capture):
        # Variable annotation syntax is only available in Python >= 3.6,
        # so we cannot declare an envelope property here
        TileServer.run_tileserver(self, self.envelope)  # type: ignore
        mp = TileServer.folium_client(self, self.envelope, capture=capture)  # type: ignore
        return mp._repr_html_()

    def _repr_html_(self):
        # These imports are here to avoid cyclic references
        from telluric.collections import BaseCollection
        from telluric.features import GeoFeature
        if (
            isinstance(self, BaseCollection) and
            self[0].has_raster
        ):
            return self._run_in_tileserver(capture="Feature collection of rasters")
        elif (
            isinstance(self, GeoFeature) and
            self.has_raster
        ):
            return self._run_in_tileserver(capture="GeoFeature with raster")

        warnings.warn(
            "Plotting a limited representation of the data, use the .plot() method for further customization")
        return simple_plot(self)._repr_html_()

    def plot(self, mp=None, **plot_kwargs):
        return plot(self, mp, **plot_kwargs)
