telluric 0.3.0 (2018-09-20)
===========================

New features
------------

* New class :py:class:`~telluric.features.GeoFeatureWithRaster` that extends
  :py:class:`~telluric.features.GeoFeature`.

telluric 0.2.1 (2018-09-12)
===========================

Bug fixes
---------

* Retrieve mask in a safer way in :py:meth:`telluric.georaster.GeoRaster2.save` (#136)
* Fix affine calculation in :py:meth:`telluric.georaster.GeoRaster2.get_tile` (#137)
* Convert dimensions to ints (#140)
* Masking areas outside the window in
  :py:meth:`telluric.georaster.GeoRaster2.get_window` (#141)
* :py:func:`telluric.georaster.merge_all` does not crash for resolution
  in ROI units (#143, #146)
* Limit rasterio version to <1.0.3
* Add LICENSE into the MANIFEST (#147)

telluric 0.2.0 (2018-08-22)
===========================

New features
------------

* Slicing a :py:class:`~telluric.collections.FeatureCollection` now returns a
  :code:`FeatureCollection` (#29, #32)
* Rasterization methods can now accept multiple fill values to produce nonbinary
  images (#34)
* :py:meth:`telluric.collections.FileCollection.save` now saves types
  better (#20, #36)
* Merging functions and :py:meth:`telluric.georaster.GeoRaster2.empty_from_roi`
  now support more ways to define the raster extent (#39, #57)
* Added utilities to convert to Cloud Optimized GeoTIFF (COG) and reproject
  files on disk (#45, #87)
* Raster data can be converted from/to different floating point formats thanks
  to enhancements in :py:meth:`telluric.georaster.GeoRaster2.astype` (#33, #66)
* Added new method :py:meth:`telluric.georaster.GeoRaster2.colorize` to colorize
  a band of a raster for visualization purposes (#81)
* Collections now have experimental "groupby/dissolve" functionality inspired
  by pandas and GeoPandas (#77, #98)
* Add a :py:data:`telluric.georaster.PixelStrategy` enum with a new mode that
  allows the user to produce the "metadata" of a merge process (#68, #91)
* :py:meth:`telluric.vectors.GeoVector.rasterize` can now accept a custom output
  CRS (#125)
* A new argument was added to the :py:class:`~telluric.vectors.GeoVector` constructor
  for disabling arguments validity checking (#126)
* Unnecessary CRS equality checking in
  :py:meth:`telluric.vectors.GeoVector.get_shape` was removed for performance
  reasons (#127)

Deprecations and removals
-------------------------

* Rasterization methods no longer support specifying a "nodata" value, and
  an appropriate nodata value will be generated
  depending on the fill value(s) (#28, #34)
* Properties in the sense of the GeoJSON standard are now called "properties"
  instead of "attributes" for consistency (#84)
* Non georeferenced raster data is no longer supported (although we are considering
  re adding it under some restrictions) (#64, #74)
* It is not required for collections to be reprojected to output CRS for
  rasterization with `fill_value` (#125)

Bug fixes
---------

* :py:meth:`telluric.vectors.GeoVector.from_record` now treats
  :code:`None` values properly (#37, #38)
* :py:class:`~telluric.georaster.GeoRaster2` methods and functions work with
  non isotropic resolution (#39)
* Cropping now behaves correctly with rasterio 1.0.0 (#44, #46)
* Crop size is now correctly computed for rasters in WGS84 (#61, #62)
* Fix rasterio 1.0.0 warnings regarding CRS comparison (#64, #74)
* :py:func:`telluric.georaster.merge_all` now is order independent and produces
  consistent results in all situations (#65, #62)
* :py:class:`~telluric.georaster.GeoRaster2` methods and functions work with
  rasters with positive y scale (#76, #78)
* :py:meth:`telluric.georaster.GeoRaster2.save` with default arguments does not
  crash for small rasters anymore (#16, #53)
* :py:meth:`telluric.collections.FileCollection.save` does not have side effects
  on heterogeneous collections anymore (#19, #24)
* Fix rasterization of points with default arguments (#9)

telluric 0.1.0 (2018-04-21)
===========================

Initial release 🎉
