import telluric


def test_api_exposed():
    assert hasattr(telluric, "GeoVector")
    assert hasattr(telluric, "GeoRaster2")
    assert hasattr(telluric, "GeoFeature")
    assert hasattr(telluric, "FeatureCollection")
