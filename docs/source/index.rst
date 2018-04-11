.. Telluric documentation master file, created by
   sphinx-quickstart on Thu Jul 27 11:04:22 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Telluric's documentation!
====================================

telluric is a Python library to manage vector and raster geospatial data in an interactive
and easy way.

The `source code`_ and `issue tracker`_ are hosted on GitHub, and all contributions and
feedback are more than welcome.

.. _`source code`: https://github.com/satellogic/telluric
.. _`issue tracker`: https://github.com/satellogic/telluric/issues

Installation
------------

You can install telluric using pip::

  pip install telluric

telluric is a pure Python library, and therefore should work on Linux, OS X and Windows
provided that you can install its dependencies. If you find any problem,
`please open an issue`_ and we will take care of it.

.. _`please open an issue`: https://github.com/satellogic/telluric/issues/new

.. warning::

    It is recommended that you **never ever use sudo** with pip because you might
    seriously break your system. Use `venv`_, `Pipenv`_, `pyenv`_ or `conda`_
    to create an isolated development environment instead.

.. _`venv`: https://docs.python.org/3/library/venv.html
.. _`Pipenv`: https://docs.pipenv.org/
.. _`pyenv`: https://github.com/pyenv/pyenv
.. _`conda`: https://conda.io/docs/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   User Guide.ipynb
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
