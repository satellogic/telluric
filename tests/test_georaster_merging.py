import pytest
from telluric import GeoRaster2

from telluric.georaster import merge_two, MergeStrategy


raster1 = GeoRaster2.open("tests/data/raster/overlap1.tif")
raster2 = GeoRaster2.open("tests/data/raster/overlap2.tif")


def test_merge_raises_error_when_rasters_do_not_overlap():
    raster1_cropped = raster1[:10, :10]
    raster2_cropped = raster2[:10, :10]

    with pytest.raises(ValueError) as excinfo:
        merge_two(raster1_cropped, raster2_cropped)

    assert "rasters do not intersect" in excinfo.exconly()


def test_merge_left_all():
    merge_strategy = MergeStrategy.LEFT_ALL

    result_left_all = merge_two(
        raster1, raster2,
        merge_strategy=merge_strategy
    )

    assert result_left_all.band_names == raster1.band_names == raster2.band_names
    assert result_left_all.footprint() == raster1.footprint()
    assert result_left_all.res_xy() == raster1.res_xy()

    result_left_all_fewer_bands = merge_two(
        raster1.limit_to_bands(["red", "green"]), raster2,
        merge_strategy=merge_strategy
    )

    assert result_left_all_fewer_bands.band_names == ["red", "green"]
    assert result_left_all_fewer_bands.footprint() == raster1.footprint()
    assert result_left_all_fewer_bands.res_xy() == raster1.res_xy()

    raster1_limited = raster1.limit_to_bands(["red", "green"])
    result_only_left = merge_two(
        raster1_limited, raster2.limit_to_bands(["blue"]),
        merge_strategy=merge_strategy
    )

    assert result_only_left == raster1_limited

    result_only_left = merge_two(
        raster1_limited, raster2.limit_to_bands(["green", "blue"]),
        merge_strategy=merge_strategy
    )

    assert result_only_left == raster1_limited


def test_merge_intersection():
    merge_strategy = MergeStrategy.INTERSECTION

    result_left_all = merge_two(
        raster1, raster2,
        merge_strategy=merge_strategy
    )

    assert result_left_all.band_names == raster1.band_names == raster2.band_names
    assert result_left_all.footprint() == raster1.footprint()
    assert result_left_all.res_xy() == raster1.res_xy()

    result_left_all_fewer_bands = merge_two(
        raster1.limit_to_bands(["red"]), raster2,
        merge_strategy=MergeStrategy.INTERSECTION
    )

    assert result_left_all_fewer_bands.band_names == ["red"]
    assert result_left_all_fewer_bands.footprint() == raster1.footprint()
    assert result_left_all_fewer_bands.res_xy() == raster1.res_xy()

    result_left_all_fewer_bands = merge_two(
        raster1, raster2.limit_to_bands(["red"]),
        merge_strategy=MergeStrategy.INTERSECTION
    )

    assert result_left_all_fewer_bands.band_names == ["red"]
    assert result_left_all_fewer_bands.footprint() == raster1.footprint()
    assert result_left_all_fewer_bands.res_xy() == raster1.res_xy()

    result_left_all_fewer_bands = merge_two(
        raster1.limit_to_bands(["red", "green"]), raster2,
        merge_strategy=MergeStrategy.INTERSECTION
    )

    assert result_left_all_fewer_bands.band_names == ["red", "green"]
    assert result_left_all_fewer_bands.footprint() == raster1.footprint()
    assert result_left_all_fewer_bands.res_xy() == raster1.res_xy()

    with pytest.raises(ValueError) as excinfo:
        merge_two(
            raster1.limit_to_bands(["red", "green"]), raster2.limit_to_bands(["blue"]),
            merge_strategy=merge_strategy
        )

    assert "rasters have no bands in common, use another merge strategy" in excinfo.exconly()


def test_merge_union():
    merge_strategy = MergeStrategy.UNION

    result_union = merge_two(
        raster1, raster2,
        merge_strategy=merge_strategy
    )

    assert result_union.band_names == raster1.band_names == raster2.band_names
    assert result_union.footprint() == raster1.footprint()
    assert result_union.res_xy() == raster1.res_xy()

    result_union_fewer_bands = merge_two(
        raster1.limit_to_bands(["red", "green"]), raster2,
        merge_strategy=merge_strategy
    )

    assert result_union_fewer_bands.band_names == ["red", "green", "blue"]
    assert result_union_fewer_bands.footprint() == raster1.footprint()
    assert result_union_fewer_bands.res_xy() == raster1.res_xy()

    result_no_common = merge_two(
        raster1.limit_to_bands(["red", "green"]), raster2.limit_to_bands(["blue"]),
        merge_strategy=merge_strategy
    )

    assert result_no_common.band_names == ["red", "green", "blue"]
