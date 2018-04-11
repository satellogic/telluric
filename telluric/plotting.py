"""Code for interactive vector plots.

"""
import warnings
from copy import copy
from statistics import median_low

import mercantile
from shapely.geometry import mapping

import folium
from ipyleaflet import (
    Map, GeoJSON,
    basemaps
)

from telluric.constants import WGS84_CRS


def simple_plot(feature, *, mp=None, **map_kwargs):
    """Plots a GeoVector in a simple Folium map.

    For more complex and customizable plots using Jupyter widgets,
    use the plot function instead.

    Parameters
    ----------
    feature : Any object with __geo_interface__
        Data to plot.

    """
    if mp is None:
        mp = folium.Map(tiles="Stamen Terrain", **map_kwargs)

    if feature.is_empty:
        warnings.warn("The geometry is empty.")

    else:
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

    if hasattr(element, "attributes") and style_func is not None:
        new_attributes = copy(element.attributes)
        new_attributes["style"] = style_func(mapping(element))
        return GeoFeature(element.geometry, new_attributes)

    else:
        # If there are no attributes or no function, there's nothing to style
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
    feature : Any object with __geo_interface__
        Data to plot.
    mp : ipyleaflet.Map, optional
        Map in which to plot, default to None (creates a new one).
    style_function : func
        Function that returns an style dictionary for
    map_kwargs : kwargs, optional
        Extra parameters to send to folium.Map.

    """
    if feature.is_empty:
        warnings.warn("The geometry is empty.")
        mp = Map(basemap=basemaps.Stamen.Terrain, **map_kwargs) if mp is None else mp

    else:
        if mp is None:
            center = feature.envelope.centroid.reproject(WGS84_CRS)
            zoom = zoom_level_from_geometry(feature.envelope)

            mp = Map(center=(center.y, center.x), zoom=zoom, basemap=basemaps.Stamen.Terrain, **map_kwargs)

        mp.add_layer(layer_from_element(feature, style_function))

    return mp


class NotebookPlottingMixin:
    def _repr_html_(self):
        warnings.warn(
            "Plotting a limited representation of the data, use the .plot() method for further customization")
        return simple_plot(self)._repr_html_()

    def plot(self, mp=None, **plot_kwargs):
        return plot(self, mp, **plot_kwargs)
