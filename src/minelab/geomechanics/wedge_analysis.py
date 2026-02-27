"""Kinematic analysis and wedge stability for rock slopes.

Planar sliding, wedge failure, toppling, Markland test, wedge factor of
safety, and stereonet data preparation.
"""

from __future__ import annotations

import numpy as np

from minelab.utilities.validators import validate_positive, validate_range

# ---------------------------------------------------------------------------
# Kinematic Planar Sliding (P3-M23)
# ---------------------------------------------------------------------------


def kinematic_planar(
    dip_face: float,
    dip_plane: float,
    friction_angle: float,
) -> dict:
    """Kinematic check for planar sliding failure.

    Planar sliding occurs when:
    1. The discontinuity dips towards the face (daylight condition).
    2. The dip of the discontinuity exceeds the friction angle.
    3. The dip of the discontinuity is less than the dip of the slope face.

    Parameters
    ----------
    dip_face : float
        Dip of the slope face (degrees, 0-90).
    dip_plane : float
        Dip of the potential sliding plane (degrees, 0-90).
    friction_angle : float
        Friction angle of the discontinuity (degrees, 0-90).

    Returns
    -------
    dict
        Keys: ``"unstable"`` (bool), ``"conditions"`` (dict of individual
        checks), ``"message"`` (str).

    Examples
    --------
    >>> result = kinematic_planar(70, 45, 35)
    >>> result["unstable"]
    True

    References
    ----------
    .. [1] Wyllie, D.C. & Mah, C.W. (2004). Rock Slope Engineering.
       4th ed., Spon Press.
    """
    validate_range(dip_face, 0, 90, "dip_face")
    validate_range(dip_plane, 0, 90, "dip_plane")
    validate_range(friction_angle, 0, 90, "friction_angle")

    # Conditions for planar sliding
    daylight = dip_plane < dip_face  # plane daylights in slope face
    exceeds_friction = dip_plane > friction_angle  # can slide

    unstable = daylight and exceeds_friction

    conditions = {
        "daylight": daylight,
        "exceeds_friction": exceeds_friction,
    }

    if unstable:
        msg = "Planar sliding kinematically possible"
    elif not daylight:
        msg = "No daylight — plane does not emerge from face"
    else:
        msg = "Stable — dip angle less than friction angle"

    return {"unstable": unstable, "conditions": conditions, "message": msg}


# ---------------------------------------------------------------------------
# Kinematic Wedge (P3-M24)
# ---------------------------------------------------------------------------


def kinematic_wedge(
    dip_face: float,
    dip_dir_face: float,
    plane1: tuple[float, float],
    plane2: tuple[float, float],
    friction_angle: float,
) -> dict:
    """Kinematic check for wedge failure.

    A wedge forms at the intersection of two discontinuity planes. Failure
    occurs when the line of intersection plunges into the slope face at
    an angle greater than the friction angle.

    Parameters
    ----------
    dip_face : float
        Dip of the slope face (degrees).
    dip_dir_face : float
        Dip direction of the slope face (degrees).
    plane1 : tuple of (float, float)
        (dip, dip_direction) of first plane.
    plane2 : tuple of (float, float)
        (dip, dip_direction) of second plane.
    friction_angle : float
        Friction angle (degrees).

    Returns
    -------
    dict
        Keys: ``"unstable"`` (bool), ``"plunge"`` (degrees),
        ``"trend"`` (degrees), ``"message"`` (str).

    Examples
    --------
    >>> result = kinematic_wedge(70, 180, (45, 160), (50, 200), 35)
    >>> isinstance(result["unstable"], bool)
    True

    References
    ----------
    .. [1] Hoek, E. & Bray, J.W. (1981). Rock Slope Engineering.
       3rd ed., IMM.
    """
    validate_range(dip_face, 0, 90, "dip_face")
    validate_range(friction_angle, 0, 90, "friction_angle")

    # Convert planes to direction cosines (pole vectors)
    def _to_normal(dip, dip_dir):
        dip_r = np.radians(dip)
        dd_r = np.radians(dip_dir)
        nx = np.sin(dip_r) * np.sin(dd_r)
        ny = np.sin(dip_r) * np.cos(dd_r)
        nz = np.cos(dip_r)
        return np.array([nx, ny, nz])

    n1 = _to_normal(plane1[0], plane1[1])
    n2 = _to_normal(plane2[0], plane2[1])

    # Line of intersection = cross product of normals
    line = np.cross(n1, n2)
    magnitude = np.linalg.norm(line)

    if magnitude < 1e-10:
        return {
            "unstable": False,
            "plunge": 0.0,
            "trend": 0.0,
            "message": "Planes are parallel — no wedge formed",
        }

    line = line / magnitude

    # Ensure line points downward
    if line[2] > 0:
        line = -line

    # Plunge and trend of line of intersection
    plunge = np.degrees(np.arcsin(-line[2]))
    trend = np.degrees(np.arctan2(line[0], line[1])) % 360

    # Check if line of intersection plunges into slope face
    daylight = plunge < dip_face
    exceeds_friction = plunge > friction_angle

    # Check trend is within face direction (±90°)
    angle_diff = abs(((trend - dip_dir_face) + 180) % 360 - 180)
    faces_slope = angle_diff < 90

    unstable = bool(daylight and exceeds_friction and faces_slope)

    msg = "Wedge failure kinematically possible" if unstable else "Wedge stable"

    return {
        "unstable": unstable,
        "plunge": float(plunge),
        "trend": float(trend),
        "message": msg,
    }


# ---------------------------------------------------------------------------
# Kinematic Toppling (P3-M25)
# ---------------------------------------------------------------------------


def kinematic_toppling(
    dip_face: float,
    dip_planes: float,
    friction_angle: float,
) -> dict:
    """Kinematic check for toppling failure.

    Toppling occurs when steeply dipping planes strike sub-parallel to
    the slope face and dip into the slope. The condition is:
    ``(90 - dip_face) + friction_angle < dip_planes``.

    Parameters
    ----------
    dip_face : float
        Dip of the slope face (degrees, 0-90).
    dip_planes : float
        Dip of the steeply dipping discontinuity set (degrees, 0-90).
    friction_angle : float
        Base friction angle (degrees, 0-90).

    Returns
    -------
    dict
        Keys: ``"unstable"`` (bool), ``"critical_dip"`` (degrees),
        ``"message"`` (str).

    Examples
    --------
    >>> result = kinematic_toppling(70, 80, 35)
    >>> result["unstable"]
    True

    References
    ----------
    .. [1] Goodman, R.E. & Bray, J.W. (1976). "Toppling of rock slopes."
       Proc. Specialty Conf. Rock Eng. for Foundations and Slopes,
       ASCE, Vol. 2, 201-234.
    """
    validate_range(dip_face, 0, 90, "dip_face")
    validate_range(dip_planes, 0, 90, "dip_planes")
    validate_range(friction_angle, 0, 90, "friction_angle")

    # Critical dip for toppling: dip > (90 - dip_face) + φ
    critical_dip = (90 - dip_face) + friction_angle
    unstable = dip_planes > critical_dip

    if unstable:
        msg = "Toppling kinematically possible"
    else:
        msg = f"Stable — dip {dip_planes}° < critical {critical_dip:.1f}°"

    return {
        "unstable": unstable,
        "critical_dip": float(critical_dip),
        "message": msg,
    }


# ---------------------------------------------------------------------------
# Markland Test (P3-M26)
# ---------------------------------------------------------------------------


def markland_test(
    face_dip: float,
    face_dip_direction: float,
    planes: list[tuple[float, float]],
    friction_angle: float,
) -> dict:
    """Combined Markland test for all kinematic failure modes.

    Parameters
    ----------
    face_dip : float
        Dip of the slope face (degrees).
    face_dip_direction : float
        Dip direction of the slope face (degrees).
    planes : list of (dip, dip_direction) tuples
        Discontinuity planes to test.
    friction_angle : float
        Friction angle (degrees).

    Returns
    -------
    dict
        Keys: ``"planar"`` (list of plane indices at risk),
        ``"wedge"`` (list of plane-pair tuples at risk),
        ``"toppling"`` (list of plane indices at risk).

    Examples
    --------
    >>> result = markland_test(70, 180, [(45, 170), (80, 10)], 35)
    >>> isinstance(result["planar"], list)
    True

    References
    ----------
    .. [1] Markland, J.T. (1972). A useful technique for estimating the
       stability of rock slopes when the rigid wedge sliding type of failure
       is expected. Imperial College Rock Mechanics Research Report No. 19.
    """
    validate_range(face_dip, 0, 90, "face_dip")
    validate_range(friction_angle, 0, 90, "friction_angle")

    planar_risk = []
    toppling_risk = []
    wedge_risk = []

    for i, (dip, dd) in enumerate(planes):
        # Planar: plane strikes sub-parallel to face, dips toward face
        strike_diff = abs(((dd - face_dip_direction) + 180) % 360 - 180)

        if strike_diff < 20:
            # Near-parallel: check planar
            result = kinematic_planar(face_dip, dip, friction_angle)
            if result["unstable"]:
                planar_risk.append(i)

        # Toppling: back-dipping planes
        if strike_diff > 160:
            result = kinematic_toppling(face_dip, dip, friction_angle)
            if result["unstable"]:
                toppling_risk.append(i)

    # Wedge: check all pairs
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            result = kinematic_wedge(
                face_dip,
                face_dip_direction,
                planes[i],
                planes[j],
                friction_angle,
            )
            if result["unstable"]:
                wedge_risk.append((i, j))

    return {
        "planar": planar_risk,
        "wedge": wedge_risk,
        "toppling": toppling_risk,
    }


# ---------------------------------------------------------------------------
# Wedge Factor of Safety (P3-M27)
# ---------------------------------------------------------------------------


def wedge_fos(
    plane1: tuple[float, float],
    plane2: tuple[float, float],
    weight: float,
    friction1: float,
    friction2: float,
    cohesion1: float = 0.0,
    cohesion2: float = 0.0,
    area1: float = 1.0,
    area2: float = 1.0,
    water_pressure: float = 0.0,
) -> dict:
    """Factor of safety for a 3D wedge failure.

    Parameters
    ----------
    plane1 : tuple of (dip, dip_direction)
        First discontinuity plane.
    plane2 : tuple of (dip, dip_direction)
        Second discontinuity plane.
    weight : float
        Weight of the wedge block (kN).
    friction1, friction2 : float
        Friction angles for planes 1 and 2 (degrees).
    cohesion1, cohesion2 : float
        Cohesion for planes 1 and 2 (kPa). Default 0.
    area1, area2 : float
        Contact areas on planes 1 and 2 (m^2). Default 1.
    water_pressure : float
        Water pressure on planes (kPa). Default 0.

    Returns
    -------
    dict
        Keys: ``"fos"`` (factor of safety), ``"driving_force"`` (kN),
        ``"resisting_force"`` (kN).

    Examples
    --------
    >>> result = wedge_fos((45, 160), (50, 200), 1000, 35, 35)
    >>> result["fos"] > 0
    True

    References
    ----------
    .. [1] Hoek, E. & Bray, J.W. (1981). Rock Slope Engineering.
       3rd ed., IMM, Ch. 8.
    """
    validate_positive(weight, "weight")

    # Direction cosines of normals
    def _normal(dip, dd):
        dip_r = np.radians(dip)
        dd_r = np.radians(dd)
        return np.array(
            [
                np.sin(dip_r) * np.sin(dd_r),
                np.sin(dip_r) * np.cos(dd_r),
                np.cos(dip_r),
            ]
        )

    n1 = _normal(plane1[0], plane1[1])
    n2 = _normal(plane2[0], plane2[1])

    # Line of intersection
    line = np.cross(n1, n2)
    mag = np.linalg.norm(line)
    if mag < 1e-10:
        return {
            "fos": float("inf"),
            "driving_force": 0.0,
            "resisting_force": 0.0,
        }
    line = line / mag
    if line[2] > 0:
        line = -line

    # Plunge of line of intersection
    plunge_rad = np.arcsin(-line[2])

    # Weight component along line of intersection (driving force)
    driving = weight * np.sin(plunge_rad)

    # Normal forces on each plane (simplified wedge analysis)
    # Resolve weight perpendicular to each plane
    w_vec = np.array([0.0, 0.0, -weight])  # weight vector pointing down

    # Normal components
    n1_force = abs(np.dot(w_vec, n1)) - water_pressure * area1
    n2_force = abs(np.dot(w_vec, n2)) - water_pressure * area2

    n1_force = max(0, n1_force)
    n2_force = max(0, n2_force)

    # Resisting force
    resisting = (
        cohesion1 * area1
        + n1_force * np.tan(np.radians(friction1))
        + cohesion2 * area2
        + n2_force * np.tan(np.radians(friction2))
    )

    fos = resisting / driving if driving > 1e-10 else float("inf")

    return {
        "fos": float(fos),
        "driving_force": float(driving),
        "resisting_force": float(resisting),
    }


# ---------------------------------------------------------------------------
# Stereonet Data (P3-M28)
# ---------------------------------------------------------------------------


def stereonet_data(
    planes: list[tuple[float, float]],
) -> dict:
    """Convert dip/dip_direction to pole and great circle data.

    Parameters
    ----------
    planes : list of (dip, dip_direction) tuples
        Discontinuity planes.

    Returns
    -------
    dict
        Keys: ``"poles"`` (list of (plunge, trend) tuples),
        ``"great_circles"`` (list of dip/dd tuples, same as input),
        ``"pole_vectors"`` (Nx3 array of unit normal vectors).

    Examples
    --------
    >>> result = stereonet_data([(45, 90)])
    >>> plunge, trend = result["poles"][0]
    >>> round(plunge, 1)
    45.0
    >>> round(trend, 1)
    270.0

    References
    ----------
    .. [1] Priest, S.D. (1993). Discontinuity Analysis for Rock
       Engineering. Chapman & Hall.
    """
    poles = []
    pole_vectors = []

    for dip, dd in planes:
        # Pole: plunge = 90 - dip, trend = dip_direction + 180
        pole_plunge = 90 - dip
        pole_trend = (dd + 180) % 360

        poles.append((float(pole_plunge), float(pole_trend)))

        # Unit normal vector (pointing away from plane)
        dip_r = np.radians(dip)
        dd_r = np.radians(dd)
        nx = np.sin(dip_r) * np.sin(dd_r)
        ny = np.sin(dip_r) * np.cos(dd_r)
        nz = np.cos(dip_r)
        pole_vectors.append([nx, ny, nz])

    return {
        "poles": poles,
        "great_circles": [(float(d), float(dd)) for d, dd in planes],
        "pole_vectors": np.array(pole_vectors),
    }
