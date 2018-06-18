class BaseFactory(object):
    @classmethod
    def get_matchings(cls, scensor_bands_info, bands):
        return [obj.name for obj in cls.objects().values()
                if obj.fits_raster_bands(scensor_bands_info, available_bands=bands)]

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
