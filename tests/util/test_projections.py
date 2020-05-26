from shapely.geometry import Point
from affine import Affine

from telluric.constants import WGS84_CRS
from telluric.util import projections


source_crs = WGS84_CRS
source_shape = Point(0.0, 0.0)


def test_gen_transform_uses_wgs84_if_no_destination():
    transformed_shape = projections.transform(source_shape, source_crs)

    assert transformed_shape == source_shape


def test_transformation():
    tf = Affine.translation(1, 2)

    # when src_affine==dst_affine should transform to itself:
    transformed_shape = projections.transform(source_shape, source_crs, src_affine=tf, dst_affine=tf)
    assert transformed_shape == source_shape

    transformed_shape = projections.transform(source_shape, source_crs, src_affine=tf)
    assert transformed_shape == Point(-1, -2)
