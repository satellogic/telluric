telluric 0.13.5 (2021-03-16)
============================

Changes
-------

* Fix memory leak in temporal rasters creation and deletion of
  :py:meth:`telluric.georaster.GeoRaster2._as_in_memory_geotiff` (#294)

telluric 0.13.4 (2021-02-23)
============================

Changes
-------

* Set :code:`_dtype` attribute in image setter for
  :py:class:`~telluric.georaster.MutableGeoRaster` (#289)
* Set :code:`crs` as empty :code:`rasterio.crs.CRS()` instance instead of :code:`None`
  when image file has no CRS (#292)
* Make :py:meth:`telluric.georaster.GeoRaster2.resize` faster (#293)

telluric 0.13.3 (2021-02-15)
============================

Changes
-------

* Add :code:`crop` parameter to  :py:func:`telluric.georaster.merge_all`
  function (#288)

telluric 0.13.2 (2020-11-27)
============================

Changes
-------

* Fix more imports when visualization dependencies are not installed (#283)

telluric 0.13.1 (2020-11-26)
============================

Changes
-------

* Fix imports when visualization dependencies are not installed (#281)
* Remove several deprecation warnings (#281)

telluric 0.13.0 (2020-11-25)
============================

Changes
-------

* Make visualization dependencies optional (#260)

telluric 0.12.1 (2020-08-10)
============================

Bug fixes
---------

* Check if the raster's footprint intersects the tile's footprint in
  :py:meth:`telluric.georaster.GeoRaster2.get_tile` (#273)

telluric 0.12.0 (2020-08-02)
============================

New features
------------

* Preserve nodata value while saving rasters (#271)
* FileCollection created out of file-like object can be iterated (#272)

telluric 0.11.1 (2020-06-27)
============================

Bug fixes
---------

* Fix :py:meth:`telluric.collections.FileCollection.sort` (#259)
* Fix potential bug in :py:class:`~telluric.context.ThreadContext` when it is uninitialized (#259)
* Disable transformation if source CRS equals to destination (#270)

telluric 0.11.0 (2019-12-02)
============================

New features
------------

* Now :py:class:`~telluric.georaster.MutableGeoRaster` inherits :code:`nodata_value`

telluric 0.10.8 (2019-08-30)
============================

Bug fixes
---------

* Now reprojection retains nodata values

telluric 0.10.7 (2019-06-06)
============================

New features
------------

* Adding support of resources accesed through HTTP and HTTPS to VRT (#248)

Big fixes
---------

* Remove unnecessary call of :py:class:`fiona.Env` (#247)

telluric 0.10.6 (2019-05-02)
============================

New features
------------

* Creating COG with internal mask (#244)
* Removed pinning for pyproj (#245)

telluric 0.10.5 (2019-04-08)
============================

Bug fixes
---------

* Workaround to overcome impossible transformations (#241)

telluric 0.10.4 (2019-03-17)
============================

Bug fixes
---------

* Prevent image loading while copying (#235)

New features
------------

* Refactored raster join implementation (#230)
* Changed default value of "nodata" in :py:class:`~telluric.georaster.GeoRaster2`
  constructor, now it is :code:`None` (#231)
* Accelerate tests (#232)
* Added new method :py:meth:`telluric.georaster.GeoRaster2.mask_by_value` (#233)
* Added new method :py:meth:`telluric.vectors.GeoVector.from_record` (#238)
* Rasterio 1.0.21 compatibility (#239)
* Adding support to lazy resize that can use overviews if exist (#240)

telluric 0.10.3 (2019-01-10)
============================

Bug fixes
---------

* Fix :py:class:`~telluric.collections.FeatureCollection` plotting (#229)

telluric 0.10.2 (2019-01-10)
============================

New features
------------

* SpatioTemporal Asset Catalog (STAC) compatibility (#223)
* Support custom schema in :py:meth:`telluric.collections.BaseCollection.save` (#224)

Bug fixes
---------

* Preserve the original schema while using :py:meth:`telluric.collections.BaseCollection.apply`
  and :py:meth:`telluric.collections.BaseCollection.groupby` (#225)
* Better handling of an empty collections (#226)
* Remove the reference to the raster object in the asset entry (#227)
* Retrieve mask in a safer way to avoid shrunk masks (#228)

telluric 0.10.1 (2018-12-27)
============================

Bug fixes
---------

* Fix masking by :py:class:`~telluric.features.GeoFeature` (#216)
* Fix issue in :py:meth:`GeoRaster.from_asset` (#217, #220)
* :py:meth:`telluric.features.GeoFeature.envelope` returns instance of
  :py:class:`~telluric.vectors.GeoVector` (#218)
* Use local tile server for visualization of :py:class:`~telluric.features.GeoFeatureWithRaster` (#221)
* :py:meth:`telluric.georaster.GeoRaster2.mask` uses crop internally to reduce memory footprint (#219)
* :py:meth:`telluric.georaster.GeoRaster2.limit_to_bands` is lazy (#222)

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
