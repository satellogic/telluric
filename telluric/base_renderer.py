class BaseFactory(object):
    @classmethod
    def get_matchings(cls, bands, sensor_bands_info=None):
        return [obj.name for obj in cls.objects().values()
                if obj.fits_raster_bands(available_bands=bands, sensor_bands_info=sensor_bands_info)]

    @classmethod
    def objects(cls):
        raise NotImplementedError

    @classmethod
    def get_object(cls, object_name, *args, **kwargs):
        """ returns instance of the object specified by object_name """
        return cls.objects()[object_name.lower()](*args, **kwargs)

    @classmethod
    def get_class(cls, object_name):
        """ returns class of the object specified by object_name """
        return cls.objects()[object_name.lower()]
