import numpy as np


def convert_meter_to_latlon_deg(lat_deg):
    m_to_deg_lat = 9e-6
    m_to_deg_lon = 9e-6 / np.cos(np.deg2rad(lat_deg))
    return m_to_deg_lat, m_to_deg_lon


def convert_resolution_from_meters_to_deg(position_lat, gsd_metric):
    m_to_deg_lat, m_to_deg_lon = convert_meter_to_latlon_deg(position_lat)
    gsd_deg_lat = gsd_metric * m_to_deg_lat
    gsd_deg_lon = gsd_metric * m_to_deg_lon
    return gsd_deg_lon, gsd_deg_lat
