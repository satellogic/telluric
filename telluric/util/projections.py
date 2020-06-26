"""Projection utilities.

"""
from functools import partial

import pyproj
from shapely import ops

from telluric.constants import WGS84_CRS


def generate_transform(source_crs, destination_crs):
    original = pyproj.Proj(dict(source_crs), preserve_units=True)
    destination = pyproj.Proj(dict(destination_crs), preserve_units=True)

    transformation = partial(
        pyproj.transform,
        original, destination
    )

    return partial(ops.transform, transformation)


def transform(shape, source_crs, destination_crs=None, src_affine=None, dst_affine=None):
    """Transforms shape from one CRS to another.

    Parameters
    ----------
    shape : shapely.geometry.base.BaseGeometry
        Shape to transform.
    source_crs : dict or str
        Source CRS in the form of key/value pairs or proj4 string.
    destination_crs : dict or str, optional
        Destination CRS, EPSG:4326 if not given.
    src_affine: Affine, optional.
        input shape in relative to this affine
    dst_affine: Affine, optional.
        output shape in relative to this affine

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Transformed shape.

    """
    if destination_crs is None:
        destination_crs = WGS84_CRS

    if src_affine is not None:
        shape = ops.transform(lambda r, q: ~src_affine * (r, q), shape)

    if source_crs != destination_crs:
        shape = generate_transform(source_crs, destination_crs)(shape)

    if dst_affine is not None:
        shape = ops.transform(lambda r, q: dst_affine * (r, q), shape)

    return shape
