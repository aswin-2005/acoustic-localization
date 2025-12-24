import numpy as np

def unit_vector_to_angles(u):
    ux, uy, uz = u

    azimuth = np.degrees(np.arctan2(uy, ux))
    elevation = np.degrees(np.arcsin(uz))

    return azimuth, elevation

def angle_to_unit_vector(azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    return np.array([x, y, z])

