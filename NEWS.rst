telluric 0.10.0 (2018-12-21)
============================

New features
------------

* Fiona 1.8.4 and Rasterio 1.0.13 compatibility (#207, #208)
* Support multiple rasters in a single :py:class:`~telluric.features.GeoFeatureWithRaster` (#209)
* Added new method :py:meth:`telluric.vectors.GeoVector.get_bounding_box` (#213)

Bug fixes
---------

* Remove hardcoded tile server port (#205)
* The internal state of the raster is not changed while saving (#210)
* Fix :py:meth:`telluric.georaster.GeoRaster2.save` (#211)
* Fix bug in reproject (#212)
* Better handling of :py:meth:`telluric.features.GeoFeature.from_record` (#214)

telluric 0.9.1 (2018-12-14)
===========================

New features
------------

* LZW compression is used by default for creating COG rasters (#200)
* Added way to change port for local tile server (#202)

Bug fixes
---------

* Fix iterating over :py:class:`~telluric.collections.FileCollection` (#203)
* Fix fiona's GDAL environment issue (#204)

telluric 0.9.0 (2018-12-12)
===========================

New features
------------

* Added new method :py:meth:`telluric.collections.FeatureCollection.from_georasters` to
  create collections of rasters (#184)
* Visualization feature collection with rasters in Jupyter Notebook (#186)
* Added new method :py:meth:`telluric.collections.BaseCollection.apply` (#188)
* Added new method :py:meth:`telluric.georaster.GeoRaster2.from_wms` for
  creating rasters out of web services (#190, #192)
* Generalizing the process of making VRT files (#191, #193)
* Rasterio 1.0.11 compatibility (#194)
* Added new method :py:meth:`telluric.georaster.GeoRaster2.from_rasters` to
  create raster out of a list of rasters (#195)
* Added support of several domains in a single VRT file (#196)

Bug fixes
---------

* Reproject features before polygonization (#182)
* Fix :py:mod:`matplotlib.cm` call (#187)
* Fix :py:meth:`telluric.georaster.GeoRaster2.save` (#197)
* Pin minimal version of Folium (#198)
* Fix rasterio's GDAL environment issue (#201)

telluric 0.8.0 (2018-11-18)
===========================

New features
------------

* Interactive representation of rasters in Jupyter Notebook (#178)
* Fiona 1.8.1 and Rasterio 1.0.10 compatibility (#179, #180)

telluric 0.7.1 (2018-11-12)
===========================

Bug fixes
---------

* Removed :py:mod:`pyplot` import from the module level to overcome issues at
  headless environments (#177)

telluric 0.7.0 (2018-11-06)
===========================

New features
------------

* Added new method :py:meth:`telluric.georaster.GeoRaster2.chunks` for
  iterating over the chunks of the raster (#169)

Bug fixes
---------

* Workaround to overcome fiona's GDAL environment issue (#175)

telluric 0.6.0 (2018-11-05)
===========================

New features
------------

* Added :code:`resampling` parameter to  :py:func:`telluric.georaster.merge_all`
  function (#166)
* New :py:meth:`telluric.vectors.GeoVector.tiles` method for iterating
  over the tiles intersecting the bounding box of the vector (#167)
* Fiona 1.8.0 compatibility (#171)

Bug fixes
---------

* Workaround to overcome rasterio's GDAL environment issue (#174)

telluric 0.5.0 (2018-10-26)
===========================

New features
------------

* A new class :py:class:`~telluric.georaster.MutableGeoRaster` was added (#165)

telluric 0.4.1 (2018-10-23)
===========================

Bug fixes
---------

* The right way to calculate :code:`dest_resolution` in :py:func:`telluric.georaster.merge_all`
  if one is not provided (#163)
* Read mask only if it exists (#164)

telluric 0.4.0 (2018-10-19)
===========================

New features
------------

* Rasterio 1.0.3 and higher compatibility (#152)
* Non-georeferenced images may be opened by providing :code:`affine` and :code:`crs` parameters
  to :py:meth:`telluric.georaster.GeoRaster2.open` (#153)
* A new argument :code:`crs` was added to :py:meth:`telluric.collections.FileCollection.open`
  for opening vector files that dont't contain information about CRS (#156)
* A new :py:func:`telluric.util.raster_utils.build_overviews` utility was added (#158)

Bug fixes
---------

* Treat 0 as legitimate value in :py:meth:`telluric.georaster.GeoRaster2.colorize` (#160)
* Fix rasterization of an empty collection with callable :code:`fill_value` (#161)

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
