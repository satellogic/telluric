# telluric

## Overview

telluric is a Python library to manage vector and raster geospatial data in an interactive
and easy way.

[![Build Status](https://travis-ci.org/satellogic/telluric.svg?branch=master)](https://travis-ci.org/satellogic/telluric)
[![Coverage](https://codecov.io/gh/satellogic/telluric/branch/master/graph/badge.svg)](https://codecov.io/gh/satellogic/telluric)

The [source code](https://github.com/satellogic/telluric) and
[issue tracker](https://github.com/satellogic/telluric/issues) are hosted on GitHub,
and all contributions and feedback are more than welcome.

## Installation

You can install telluric using pip:

```
pip install telluric
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
