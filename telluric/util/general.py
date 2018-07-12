import numpy as np


def convert_meter_to_latlon_deg(lat_deg):
    # Scaling factor is 1 / 111.32 km, hence approximating a spherical Earth
    # See https://en.wikipedia.org/wiki/Decimal_degrees#Precision
    m_to_deg_lat = 1 / 111320.
    m_to_deg_lon = 1 / (111320. * np.cos(np.deg2rad(lat_deg)))
    return m_to_deg_lat, m_to_deg_lon


def convert_resolution_from_meters_to_deg(position_lat, gsd_metric):
    m_to_deg_lat, m_to_deg_lon = convert_meter_to_latlon_deg(position_lat)
    gsd_deg_lat = gsd_metric * m_to_deg_lat
    gsd_deg_lon = gsd_metric * m_to_deg_lon
    return gsd_deg_lon, gsd_deg_lat
