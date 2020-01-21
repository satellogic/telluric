"""Telluric thread local context manager.

Inspired by rasterio, (c) MapBox, MIT License
"""
import logging
import threading


class ThreadContext(threading.local):
    def __init__(self):
        # Initializes in each thread
        self._options = {}

    def get(self, key, default=None):
        return self._options.get(key, default)


local_context = ThreadContext()

log = logging.getLogger(__name__)


class TelluricContextError(Exception):
    pass


class TelluricContext(object):
    """TelluricContext is a thread.local context manager for telluric parameters,
    that can be used in telluric functionalities.

    Example when applying a product on a raster for setting sensor_bands_info:
        with TelluricContext(sensor_bands_info=bands_mapping):
            return raster.apply('NDVI')

    The arguments are saved in the `context` module under `local_contex` and to read them just use
    `local_context.get("argument_name"), for example:

        from telluric.context import local_context, TelluricContext

        with TelluricContex('a'=11):
            local_context.get('a')  # => 11
            local_context.get('b')  # => None
            local_context.get('c', 44)  # => 44

    TelluricContext can be nested and in that case the inner context overides the outer, for example:
        from telluric.context import local_context, TelluricContext

        with TelluricContex('a'=11):
            with TelluricContex('a'=33, 'b'='some_str'):
                local_context.get('a')  # => 33
                local_context.get('b')  # => some_str
            local_context.get('a')  # => 11
            local_context.get('b')  # => None
    """

    @classmethod
    def default_options(cls):
        """Default configuration options
        Parameters
        ----------
        None
        Returns
        -------
        dict
        """
        return {
        }

    def __init__(
            self, **options):
        """Create a new telluricContext settings.
        Parameters
        ----------
        **options : optional
            A mapping of key:value options, e.g.,
            `CPL_DEBUG=True, CHECK_WITH_INVERT_PROJ=False`.

        Returns
        -------
        TelluricContext
        """
        self.options = options.copy()
        self.context_options = {}  # type: dict

    @classmethod
    def from_defaults(cls, **kwargs):
        """Create a context with default config options
        Parameters
        ----------
        kwargs : optional
            Keyword arguments for TelluricContext()
        Returns
        -------
        TelluricContext
        Notes
        -----
        The items in kwargs will be overlaid on the default values.
        """
        options = TelluricContext.default_options()
        options.update(**kwargs)
        return cls(**options)

    def __enter__(self):
        log.debug("Entering env context: %r", self)
        # No parent TelluricContext exists.
        if not local_context._options:
            log.debug("Starting outermost env")
            self._has_parent_env = False

            reset_context(**self.options)
            self.context_options = {}
        else:
            self._has_parent_env = True
            self.context_options = get_context()
            set_context(**self.options)

        log.debug("Entered env context: %r", self)
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        log.debug("Exiting env context: %r", self)
        del_context()
        if self._has_parent_env:
            reset_context(**self.context_options)
        else:
            log.debug("Exiting outermost env")
        log.debug("Exited env context: %r", self)


def reset_context(**options):
    """Reset context to default."""
    local_context._options = {}
    local_context._options.update(options)
    log.debug("New TelluricContext context %r created", local_context._options)


def get_context():
    """Get a mapping of current options."""
    if not local_context._options:
        raise TelluricContextError("TelluricContext context not exists")
    else:
        log.debug("Got a copy of context %r options", local_context._options)
        return local_context._options.copy()


def set_context(**options):
    """Set options in the existing context."""
    if not local_context._options:
        raise TelluricContextError("TelluricContext context not exists")
    else:
        local_context._options.update(options)
        log.debug("Updated existing %r with options %r", local_context._options, options)


def del_context():
    """Delete options in the existing context."""
    if not local_context._options:
        raise TelluricContextError("TelluricContext context not exists")
    local_context._options.clear()
