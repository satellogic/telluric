# telluric

## Overview

telluric is a Python library to manage vector and raster geospatial data in an interactive
and easy way.

[![Build Status](https://travis-ci.org/satellogic/telluric.svg?branch=master)](https://travis-ci.org/satellogic/telluric)
[![Coverage](https://codecov.io/gh/satellogic/telluric/branch/master/graph/badge.svg)](https://codecov.io/gh/satellogic/telluric)
[![Chat](https://img.shields.io/matrix/telluric-dev:matrix.org.svg?style=flat-square)](https://riot.im/app/#/room/#telluric-dev:matrix.org)

Opening a raster is as simple as:

```
In [1]: import telluric as tl

In [2]: tl.GeoRaster2.open("http://download.osgeo.org/geotiff/samples/usgs/f41078e1.tif")
Out[2]: <telluric.georaster.GeoRaster2 at 0x7facd183ad68>
```

And reading some vector data is equally simple:

```
In [3]: tl.FileCollection.open("shapefiles/usa-major-cities.shp")
Out[3]: <telluric.collections.FileCollection at 0x7facd1183048>
```

For more usage examples and a complete API reference,
[check out our documentation](http://telluric.readthedocs.io/) on Read the Docs.

The [source code](https://github.com/satellogic/telluric) and
[issue tracker](https://github.com/satellogic/telluric/issues) are hosted on GitHub,
and all contributions and feedback are more than welcome.

## Installation

You can install telluric using pip:

```
pip install telluric[vis]
```

Read more complete installation instructions at [our documentation](http://telluric.readthedocs.io/).

telluric is a pure Python library, and therefore should work on Linux, OS X and Windows
provided that you can install its dependencies. If you find any problem,
[please open an issue](https://github.com/satellogic/telluric/issues/new)
and we will take care of it.

## Development

telluric is usually developed on Linux. For full tests do:

```
$ make build
$ make test
```

for testing single tests do:

```
$ make dockershell
docker$ python -m pytest TEST_FILE::TEST_NAME
```

## Support

Join our [Matrix chat](https://riot.im/app/#/room/#telluric-dev:matrix.org) to ask all sorts of questions!
