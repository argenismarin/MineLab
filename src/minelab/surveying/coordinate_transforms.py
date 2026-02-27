"""Coordinate transformation utilities for mine surveying.

This module provides functions for UTM/lat-lon conversions, local mine
coordinate transforms, drillhole desurvey, and bearing-distance
calculations.  All conversions use simplified formulas suitable for
mining accuracy (approximately +/-1 m).

References
----------
.. [1] Karney, C.F.F. (2011). *Transverse Mercator with an accuracy of a
       few nanometers*. J. Geodesy, 85(8), 475--485.
.. [2] Snyder, J.P. (1987). *Map Projections -- A Working Manual*. USGS
       Professional Paper 1395.
"""

from __future__ import annotations

import math

from minelab.utilities.validators import (
    validate_non_negative,
    validate_positive,
    validate_range,
)

# WGS-84 ellipsoid parameters
_A = 6378137.0  # semi-major axis (m)
_F = 1.0 / 298.257223563  # flattening
_B = _A * (1.0 - _F)  # semi-minor axis
_E2 = 2.0 * _F - _F**2  # first eccentricity squared
_E_PRIME2 = _E2 / (1.0 - _E2)  # second eccentricity squared
_K0 = 0.9996  # UTM scale factor
_FALSE_EASTING = 500000.0
_FALSE_NORTHING_S = 10000000.0  # false northing for southern hemisphere


# ---------------------------------------------------------------------------
# UTM to Lat/Lon
# ---------------------------------------------------------------------------


def utm_to_latlon(
    easting: float,
    northing: float,
    zone_number: int,
    zone_letter: str,
) -> dict:
    """Convert UTM coordinates to latitude and longitude (WGS-84).

    Implements a simplified inverse transverse Mercator projection
    suitable for mining survey accuracy (~1 m).

    Parameters
    ----------
    easting : float
        UTM easting in metres.  Must be positive.
    northing : float
        UTM northing in metres.  Must be non-negative.
    zone_number : int
        UTM zone number (1--60).
    zone_letter : str
        UTM zone letter (C--X, excluding I and O).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"latitude"`` : float -- Latitude in decimal degrees.
        - ``"longitude"`` : float -- Longitude in decimal degrees.

    Examples
    --------
    >>> r = utm_to_latlon(500000, 0, 31, 'N')
    >>> round(r["latitude"], 4)
    0.0
    >>> round(r["longitude"], 4)
    3.0

    References
    ----------
    .. [1] Karney (2011).
    .. [2] Snyder (1987), Ch. 8.
    """
    validate_positive(easting, "easting")
    validate_non_negative(northing, "northing")
    validate_range(zone_number, 1, 60, "zone_number")

    zone_letter = zone_letter.upper()
    if zone_letter not in "CDEFGHJKLMNPQRSTUVWX":
        raise ValueError(f"'zone_letter' must be C-X (excluding I, O), got '{zone_letter}'.")

    hemisphere_north = zone_letter >= "N"
    central_meridian = (zone_number - 1) * 6 - 180 + 3

    x = easting - _FALSE_EASTING
    y = northing
    if not hemisphere_north:
        y = y - _FALSE_NORTHING_S

    # Footprint latitude
    m = y / _K0
    mu = m / (_A * (1.0 - _E2 / 4.0 - 3.0 * _E2**2 / 64.0 - 5.0 * _E2**3 / 256.0))

    e1 = (1.0 - math.sqrt(1.0 - _E2)) / (1.0 + math.sqrt(1.0 - _E2))

    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
    )

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)
    n1 = _A / math.sqrt(1.0 - _E2 * sin_phi1**2)
    t1 = tan_phi1**2
    c1 = _E_PRIME2 * cos_phi1**2
    r1 = _A * (1.0 - _E2) / ((1.0 - _E2 * sin_phi1**2) ** 1.5)
    d = x / (n1 * _K0)

    lat = phi1 - (n1 * tan_phi1 / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * _E_PRIME2) * d**4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1**2 - 252.0 * _E_PRIME2 - 3.0 * c1**2)
        * d**6
        / 720.0
    )

    lon = (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * _E_PRIME2 + 24.0 * t1**2)
        * d**5
        / 120.0
    ) / cos_phi1

    latitude = math.degrees(lat)
    longitude = math.degrees(lon) + central_meridian

    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
    }


# ---------------------------------------------------------------------------
# Lat/Lon to UTM
# ---------------------------------------------------------------------------


def latlon_to_utm(
    latitude: float,
    longitude: float,
) -> dict:
    """Convert latitude and longitude (WGS-84) to UTM coordinates.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees.  Must be in [-84, 84].
    longitude : float
        Longitude in decimal degrees.  Must be in [-180, 180].

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"easting"`` : float -- UTM easting in metres.
        - ``"northing"`` : float -- UTM northing in metres.
        - ``"zone_number"`` : int -- UTM zone number.
        - ``"zone_letter"`` : str -- UTM zone letter.

    Examples
    --------
    >>> r = latlon_to_utm(0.0, 3.0)
    >>> round(r["easting"], 0)
    500000.0
    >>> r["zone_number"]
    31

    References
    ----------
    .. [1] Snyder (1987), Ch. 8.
    """
    validate_range(latitude, -84, 84, "latitude")
    validate_range(longitude, -180, 180, "longitude")

    zone_number = int(math.floor((longitude + 180.0) / 6.0)) + 1

    # Zone letter
    letters = "CDEFGHJKLMNPQRSTUVWX"
    lat_idx = int((latitude + 80.0) / 8.0)
    lat_idx = max(0, min(lat_idx, len(letters) - 1))
    zone_letter = letters[lat_idx]

    central_meridian = (zone_number - 1) * 6 - 180 + 3
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude - central_meridian)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)

    n = _A / math.sqrt(1.0 - _E2 * sin_lat**2)
    t = tan_lat**2
    c = _E_PRIME2 * cos_lat**2
    a_coeff = cos_lat * lon_rad

    m = _A * (
        (1.0 - _E2 / 4.0 - 3.0 * _E2**2 / 64.0 - 5.0 * _E2**3 / 256.0) * lat_rad
        - (3.0 * _E2 / 8.0 + 3.0 * _E2**2 / 32.0 + 45.0 * _E2**3 / 1024.0)
        * math.sin(2.0 * lat_rad)
        + (15.0 * _E2**2 / 256.0 + 45.0 * _E2**3 / 1024.0) * math.sin(4.0 * lat_rad)
        - (35.0 * _E2**3 / 3072.0) * math.sin(6.0 * lat_rad)
    )

    easting = (
        _K0
        * n
        * (
            a_coeff
            + (1.0 - t + c) * a_coeff**3 / 6.0
            + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * _E_PRIME2) * a_coeff**5 / 120.0
        )
        + _FALSE_EASTING
    )

    northing = _K0 * (
        m
        + n
        * tan_lat
        * (
            a_coeff**2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c**2) * a_coeff**4 / 24.0
            + (61.0 - 58.0 * t + t**2 + 600.0 * c - 330.0 * _E_PRIME2) * a_coeff**6 / 720.0
        )
    )

    if latitude < 0:
        northing += _FALSE_NORTHING_S

    return {
        "easting": float(easting),
        "northing": float(northing),
        "zone_number": int(zone_number),
        "zone_letter": str(zone_letter),
    }


# ---------------------------------------------------------------------------
# Grid to Mine Coordinates
# ---------------------------------------------------------------------------


def grid_to_mine_coordinates(
    easting: float,
    northing: float,
    origin_e: float,
    origin_n: float,
    rotation_deg: float,
) -> dict:
    """Transform grid coordinates to local mine coordinates.

    Translates and rotates grid coordinates relative to a mine origin
    and rotation angle.

    .. math::

        \\Delta E &= E - E_0

        \\Delta N &= N - N_0

        E_{mine} &= \\Delta E \\cos\\theta + \\Delta N \\sin\\theta

        N_{mine} &= -\\Delta E \\sin\\theta + \\Delta N \\cos\\theta

    Parameters
    ----------
    easting : float
        Grid easting in metres.
    northing : float
        Grid northing in metres.
    origin_e : float
        Mine origin easting in metres.
    origin_n : float
        Mine origin northing in metres.
    rotation_deg : float
        Rotation angle in degrees (clockwise from grid north).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"mine_easting"`` : float
        - ``"mine_northing"`` : float

    Examples
    --------
    >>> r = grid_to_mine_coordinates(1100, 2050, 1000, 2000, 0)
    >>> r["mine_easting"]
    100.0
    >>> r["mine_northing"]
    50.0

    References
    ----------
    .. [1] Uren & Price (2010), Ch. 6.
    """
    rot_rad = math.radians(rotation_deg)
    dx = easting - origin_e
    dy = northing - origin_n

    mine_e = dx * math.cos(rot_rad) + dy * math.sin(rot_rad)
    mine_n = -dx * math.sin(rot_rad) + dy * math.cos(rot_rad)

    return {
        "mine_easting": float(mine_e),
        "mine_northing": float(mine_n),
    }


# ---------------------------------------------------------------------------
# Collar to Downhole (Simple Tangential)
# ---------------------------------------------------------------------------


def collar_to_downhole(
    x0: float,
    y0: float,
    z0: float,
    azimuth: float,
    dip: float,
    depth_intervals: list,
) -> dict:
    """Compute downhole coordinates using the simple tangential method.

    For each depth interval the displacement from the collar is:

    .. math::

        \\Delta x &= d \\sin(\\text{az}) \\cos(\\text{dip})

        \\Delta y &= d \\cos(\\text{az}) \\cos(\\text{dip})

        \\Delta z &= -d \\sin(\\text{dip})

    where dip is negative for downward.

    Parameters
    ----------
    x0 : float
        Collar X coordinate.
    y0 : float
        Collar Y coordinate.
    z0 : float
        Collar Z (elevation) coordinate.
    azimuth : float
        Hole azimuth in degrees from north (0--360).
    dip : float
        Hole dip in degrees (negative = downward, typically
        -90 for vertical).  Must be in [-90, 90].
    depth_intervals : list of float
        Cumulative depths from collar at which to compute
        coordinates.  Must all be non-negative.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"x"`` : list of float
        - ``"y"`` : list of float
        - ``"z"`` : list of float
        - ``"total_depth"`` : float

    Examples
    --------
    >>> r = collar_to_downhole(1000, 2000, 500, 0, -90, [0, 50, 100])
    >>> r["z"][-1]
    600.0

    References
    ----------
    .. [1] Uren & Price (2010), Ch. 22.
    """
    validate_range(dip, -90, 90, "dip")
    if not depth_intervals:
        raise ValueError("'depth_intervals' must contain at least one value.")
    for i, d in enumerate(depth_intervals):
        if d < 0:
            raise ValueError(f"'depth_intervals[{i}]' must be non-negative, got {d}.")

    az_rad = math.radians(azimuth)
    dip_rad = math.radians(dip)

    xs = []
    ys = []
    zs = []

    for d in depth_intervals:
        dx = d * math.sin(az_rad) * math.cos(dip_rad)
        dy = d * math.cos(az_rad) * math.cos(dip_rad)
        dz = -d * math.sin(dip_rad)
        xs.append(float(x0 + dx))
        ys.append(float(y0 + dy))
        zs.append(float(z0 + dz))

    total_depth = float(max(depth_intervals)) if depth_intervals else 0.0

    return {
        "x": xs,
        "y": ys,
        "z": zs,
        "total_depth": total_depth,
    }


# ---------------------------------------------------------------------------
# Bearing and Distance
# ---------------------------------------------------------------------------


def bearing_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> dict:
    """Compute bearing and distance between two points.

    .. math::

        \\theta &= \\text{atan2}(\\Delta x,\\; \\Delta y)

        d &= \\sqrt{\\Delta x^2 + \\Delta y^2}

    Bearing is measured from north (0 deg) clockwise.

    Parameters
    ----------
    x1 : float
        X (easting) of first point.
    y1 : float
        Y (northing) of first point.
    x2 : float
        X (easting) of second point.
    y2 : float
        Y (northing) of second point.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"bearing_deg"`` : float -- Bearing in degrees [0, 360).
        - ``"distance_m"`` : float -- Euclidean distance in metres.

    Examples
    --------
    >>> r = bearing_distance(0, 0, 100, 100)
    >>> round(r["bearing_deg"], 1)
    45.0
    >>> round(r["distance_m"], 2)
    141.42

    References
    ----------
    .. [1] Uren & Price (2010), Ch. 6.
    """
    dx = x2 - x1
    dy = y2 - y1

    distance = math.sqrt(dx**2 + dy**2)
    bearing = math.degrees(math.atan2(dx, dy))
    if bearing < 0:
        bearing += 360.0

    return {
        "bearing_deg": float(bearing),
        "distance_m": float(distance),
    }
