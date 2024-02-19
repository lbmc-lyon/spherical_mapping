import numpy as np
from collections.abc import Iterable
from scipy.spatial.transform import Rotation


def cart2spher(coords: Iterable):
    """
    Get spherical coordinate from cartesian coordinates. Careful : the angles corresponds to longitude and latitude.
    :param coords: List of (x, y, z)
    :return: Corresponding array of (rho, theta, phi); rho : distance from origin, theta : latitude [-pi/2, pi/2],
    phi : longitude [-pi, pi].
    """
    if len(np.shape(coords)) > 1:
        x_, y_, z_ = np.transpose(coords)
    else:
        x_, y_, z_ = coords

    rho = np.sqrt(x_**2 + y_**2 + z_**2)
    theta = np.arctan2(z_, np.sqrt(x_**2 + y_**2)) * 180 / np.pi  # Latitude, not colatitude !
    phi = np.arctan2(y_, x_) * 180 / np.pi
    return np.transpose([rho, theta, phi])

def spher2cart(coords: Iterable):
    """
    Get cartesian coordinates from spherical coordinates. Careful : the order of spherical coords must corresponds to
    (distance from orig., latitude, longitude), cf. cart2spher.
    :param coords: List of (rho, theta, phi); rho : distance from origin, theta : latitude [-pi/2, pi/2],
    phi : longitude [-pi, pi].
    :return: Corresponding array of (x, y, z).
    """
    if len(np.shape(coords)) > 1:
        rho_, theta_, phi_ = np.transpose(coords)
    else:
        rho_, theta_, phi_ = coords
    theta_ *=  np.pi / 180
    theta_ = np.pi / 2 - theta_  # Convert theta to colatitude. As defined in cart2spher, theta is the latitude.
    phi_ *=  np.pi / 180

    x = rho_ * np.sin(theta_) * np.cos(phi_)
    y = rho_ * np.sin(theta_) * np.sin(phi_)
    z = rho_ * np.cos(theta_)
    return np.transpose([x, y, z])

def mask_between_angles(spher_coord: Iterable, theta_range: Iterable, phi_range: Iterable):
    """
    Get a mask of spher_coord where True means the coordinate is inside theta_range and inside phi_range.
    :param spher_coord: List of (rho, theta, phi); rho : distance from origin, theta : latitude [-pi/2, pi/2],
    phi : longitude [-pi, pi].
    :param theta_range: Latitude (theta) angles between which the coordinates are tested.
    :param phi_range: Longitude (phi) angles between which the coordinates are tested.
    :return: Mask of spher_coords of the coordinates inside theta and phi ranges given.
    """
    if len(np.shape(spher_coord)) > 1:
        rho_, theta_, phi_ = np.transpose(spher_coord)
    else:
        rho_, theta_, phi_ = spher_coord
    mask_theta = (np.array(theta_) < theta_range[1]) * (np.array(theta_) > theta_range[0])
    mask_phi = (np.array(phi_) < phi_range[1]) * (np.array(phi_) > phi_range[0])
    return mask_theta * mask_phi


def sphere_mapping(spher_coords: Iterable, val: Iterable, angle_step=10, half_window=7):
    """
    Maps and averages the values of a curved surface on a 2D plane. X-axis corresponds to the longitude, Y-axis to the
    latitude (similar to earth maps). Each sector (defined by the angle_step +/- half window for both angles) will have
    the average value of the values inside it.
    :param spher_coords: List of (rho, theta, phi); rho : distance from origin, theta : latitude [-pi/2, pi/2],
    phi : longitude [-pi, pi].
    :param val: Values corresponding to each coordinates
    :param angle_step: Step between each sector for both longitude and latitude.
    :param half_window: Half width (angle) of each sector.
    :return: List of latitude center of each sector, List of longitude center of each sector, average value in each
    sector.
    """
    theta_angles = np.arange(-90, 90, angle_step)
    phi_angles = np.arange(-180, 180, angle_step)
    mat_mask = []
    mean_vals = []
    for theta_ in theta_angles:
        mat_mask.append([])
        mean_vals.append([])
        for phi_ in phi_angles:
            mat_mask[-1].append(
                mask_between_angles(spher_coords,
                                    (theta_ - half_window, theta_ + half_window),
                                    (phi_ - half_window, phi_ + half_window)
                                    )
            )
            #TODO Speed up the process by using numpy mean() (Careful about the axis...)
            mean_vals[-1].append(np.mean(np.array(val)[mat_mask[-1][-1]]))

    mat_mask = np.array(mat_mask)

    return theta_angles, phi_angles, mean_vals

def ref_point(center_coord : Iterable, impact_coord : Iterable, node_coords : Iterable):
    """
    Translates coordinates in node_coords to the origin center_coord then rotates node_coords around the origin so that
    impact_coords is the point of zero angle (theta=0 and phi=0) in spheric coordinates.
    :param center_coord: Coordinate (x, y, z) of the new origin.
    :param impact_coord: Coordinate (x, y, z) of the point by which the spherical coordinate will be theta=0 and phi=0.
    :param node_coords: List of coordinates (x, y, z) that will be transformed.
    :return: Transformed list of coordinates (x, y, z).
    """
    node_coords_ = np.array(node_coords) - np.array(center_coord)
    impact_coords_ = np.array(impact_coord) - np.array(center_coord)
    impact_coords_spher = cart2spher(impact_coords_)
    r = Rotation.from_euler('ZYX', (impact_coords_spher[2], impact_coords_spher[1], 0), degrees=True)
    node_coords_ = r.apply(node_coords_, inverse=True)
    return node_coords_


def to_2d_map(cartesian_coords: Iterable, values: Iterable, center_coord: Iterable=(0, 0, 0),
              impact_ref_coord: Iterable=(0, 0, 0), angle_step=5, half_window=3):
    """
    Maps the nodes in a 3D space to a 2D plane with X-axis : phi, and Y-axis : theta (angles of spherical coordinates,
    cf. cart2spher() ).
    :param cartesian_coords: List of coordinates (x, y, z) to be mapped.
    :param values: Values corresponding to the coordinates to be mapped.
    :param center_coord: Coordinate (x, y, z) of the new origin (cf. ref_point() ).
    :param impact_ref_coord: Coordinate (x, y, z) of the point by which the spherical coordinate will be theta=0 and
    phi=0 (cf. ref_point() ).
    :param angle_step: Step between each sector for both longitude and latitude (cf. sphere_mapping() ).
    :param half_window: Half width (angle) of each sector (cf. sphere_mapping() ).
    :return: theta_angles: Array of theta center of each sector,
    phi_angles: Array of phi center of each sector,
    mean_vals: Array of mean values in each sector.
    """
    coords = ref_point(center_coord, impact_ref_coord, cartesian_coords)
    spher_coords = cart2spher(coords)
    theta_angles, phi_angles, mean_vals = sphere_mapping(spher_coords, values, angle_step=angle_step,
                                                         half_window=half_window)
    return theta_angles, phi_angles, mean_vals
