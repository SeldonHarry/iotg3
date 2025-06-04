import numpy as np
import sys
import math
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

# from ray_tracing_utils import get_distance_attenuation, get_euclidean_distance, check_ray_sphere_intersection, get_angle_between_vec

from PIL import Image, ImageFilter, ImageDraw, ImageChops
from scipy.ndimage import binary_erosion, iterate_structure
import time

# from ray_tracing_utils import generate_rays_3d


def vec_abs(v):
    """计算向量模长"""
    return np.sqrt(np.sum(v**2))


def vec_dot(v1, v2):
    """计算向量点积，都是 1d np array"""

    return np.sum(v1 * v2, axis=-1)


def get_angle_between_vec(v1, v2):
    """
    计算两个向量之间的夹角 0<theta<180，不分左右
    """
    # temp = vec_dot(v1, v2) / (vec_abs(v1) * vec_abs(v2))
    # print(temp)
    # if vec_abs(v1) * vec_abs(v2) == 0:
    #     raise RuntimeError("get_angle_between_vec: divide zero")
    # try:
    #     angle = np.rad2deg(np.arccos(vec_dot(v1, v2) / (vec_abs(v1) * vec_abs(v2))))
    # except Warning:
    #     print(v1)
    #     print(v2)
    #     print(vec_dot(v1, v2))
    #     print(vec_abs(v1) * vec_abs(v2))
    if vec_dot(v1, v2) / (vec_abs(v1) * vec_abs(v2)) > 1:
        return np.rad2deg(np.arccos(1.0))
    elif vec_dot(v1, v2) / (vec_abs(v1) * vec_abs(v2)) < -1:
        return np.rad2deg(np.arccos(-1.0))
    else:
        return np.rad2deg(np.arccos(vec_dot(v1, v2) / (vec_abs(v1) * vec_abs(v2))))


def check_ray_sphere_intersection(
    ray_origin, ray_direction, sphere_centroid, sphere_radius
):
    """
    Check whether a ray intersects with a sphere.
    If intersect, return the intersection point and the closest point from sphere centroid to the ray.

    Parameters:
    - ray_origin: 1D array representing the origin of the ray
    - ray_direction: 1D array representing the direction of the ray, should be a unit vector
    - sphere_centroid: 1D array representing the center of the sphere
    - sphere_radius: float representing the radius of the sphere

    Returns:
    - if_intersect: Boolean indicating if there is an intersection
    - Q: The intersection point, if any
    - M: The closest point on the ray from the sphere centroid, if any
    """
    if not (ray_origin.ndim == ray_direction.ndim == sphere_centroid.ndim == 1):
        raise ValueError("Input shape error")
    if ray_origin.size != sphere_centroid.size:
        raise ValueError("Ray and sphere are not in the same dimension")

    # Ensure ray_direction is a unit vector
    if not np.isclose(np.linalg.norm(ray_direction), 1):
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

    C = sphere_centroid
    R = sphere_radius
    d_vec = ray_direction
    P = ray_origin

    if_intersect = True

    a_vec = C - P
    l = np.dot(d_vec, a_vec)
    a_square = np.dot(a_vec, a_vec)
    if a_square > R**2 and l < 0:
        if_intersect = False

    # The closest distance from sphere centroid to ray
    m_square = a_square - l**2
    if m_square > R**2:
        if_intersect = False

    if if_intersect:
        q = np.sqrt(R**2 - m_square)
        if a_square > R**2:
            # Ray origin is outside the sphere
            t = l - q
        else:
            # Ray origin is inside the sphere
            t = l + q
        Q = P + t * d_vec  # Intersection point
        M = P + l * d_vec  # Closest point from sphere centroid to ray

        return if_intersect, Q, M
    else:
        return if_intersect, None, None


def get_euclidean_distance(p1, p2):
    """计算两点间的欧氏距离"""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def get_distance_attenuation(distance):
    """因为 2d 也是在模拟 3d，所以都按照三维空间中球的表面积来衰减"""
    absorption = np.exp(-0.01 * distance)
    if distance <= np.sqrt(1 / (4 * np.pi)):
        return 1.0 * absorption
    else:
        return (1.0 / (4.0 * np.pi * distance**2)) * absorption


def ray_surface_intersection(A, B, C, D, E, F, G, ray_origin, ray_direction):
    """
    Check if a ray intersects with a quadratic surface or a plane and find the intersection point.

    Parameters:
    - A, B, C, D, E, F, G: coefficients of the surface equation Ax^2 + By^2 + Cz^2 + Dx + Ey + Fz + G = 0.
    - ray_origin: Origin of the ray (numpy array [x0, y0, z0]).
    - ray_direction: Direction of the ray (numpy array [dx, dy, dz]).

    Returns:
    - (True, intersection point) if there is an intersection.
    - (False, None) if there is no intersection.
    """
    x0, y0, z0 = ray_origin
    dx, dy, dz = ray_direction

    if A == B == C == 0:  # The surface is a plane
        denominator = D * dx + E * dy + F * dz
        if denominator == 0:  # Ray is parallel to the plane
            return (False, None)
        t = -(D * x0 + E * y0 + F * z0 + G) / denominator
        if t < 0:  # Intersection point is behind the ray's origin
            return (False, None)
        intersection_point = ray_origin + t * ray_direction
        return (True, intersection_point)
    else:  # The surface is a quadratic surface
        a = A * dx**2 + B * dy**2 + C * dz**2
        b = (
            2 * A * x0 * dx
            + 2 * B * y0 * dy
            + 2 * C * z0 * dz
            + D * dx
            + E * dy
            + F * dz
        )
        c = A * x0**2 + B * y0**2 + C * z0**2 + D * x0 + E * y0 + F * z0 + G

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return (False, None)

        # When a is zero (for a linear equation in t), handle division by zero
        if a == 0:
            t = -c / b
            if t < 0:
                return (False, None)
            intersection_point = ray_origin + t * ray_direction
            return (True, intersection_point)

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        t_values = [t for t in [t1, t2] if t >= 0]

        if not t_values:
            return (False, None)

        t_min = min(t_values)
        intersection_point = ray_origin + t_min * ray_direction
        return (True, intersection_point)


def is_point_inside_polygon(point, polygon):
    """
    Determine if a point is inside a given polygon or not.

    Parameters:
    - point: A array (x, y) representing the point to check.
    - polygon: A list of array [(x1, y1), (x2, y2), ..., (xn, yn)] representing the vertices of the polygon.

    Returns:
    - True if the point is inside the polygon, False otherwise.
    """
    # x, y = point
    x = point[0]
    y = point[1]
    inside = False

    n = len(polygon)
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def surface_inclusion_test(p, xy_bound_points, xz_bound_points, yz_bound_points):
    """提不出边界点，说明几乎垂直于某平面，不纳入考虑范围，直接True"""

    if len(xy_bound_points) > 2:
        p_2d = p[[0, 1]]
        is_include_xy = is_point_inside_polygon(point=p_2d, polygon=xy_bound_points)
    else:
        is_include_xy = True

    if len(xz_bound_points) > 2:
        p_2d = p[[0, 2]]
        is_include_xz = is_point_inside_polygon(point=p_2d, polygon=xz_bound_points)
    else:
        is_include_xz = True

    if len(yz_bound_points) > 2:
        p_2d = p[[1, 2]]
        is_include_yz = is_point_inside_polygon(point=p_2d, polygon=yz_bound_points)
    else:
        is_include_yz = True

    if is_include_xy and is_include_xz and is_include_yz:
        return True
    else:
        return False


def get_surface_normal(A, B, C, D, E, F, point):
    """
    Calculate the normal vector of the surface A*x^2 + B*y^2 + C*z^2 + D*x + E*y + F*z + G = 0
    at a given point.

    Parameters:
    - A, B, C, D, E, F: coefficients of the surface equation.
    - point: A point (x, y, z) on the surface, given as a tuple or list.

    Returns:
    - A numpy array representing the normal vector at the given point.
    """
    x, y, z = point
    # Calculate the gradient vector (normal vector) at the point
    normal_vector = np.array([2 * A * x + D, 2 * B * y + E, 2 * C * z + F])
    # Optionally, normalize the vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector


def intersection_test(surface_list, ray_origin, ray_direction):
    # global T

    # tt = time.time()
    intersect_t_list = []
    for w_list, xy_bound_points, xz_bound_points, yz_bound_points in surface_list:
        # surface_points = surface_points_list[i]
        # tt = time.time()
        is_intersect, intersect_p = ray_surface_intersection(
            A=0,
            B=0,
            C=0,
            D=w_list[0],
            E=w_list[1],
            F=w_list[2],
            G=w_list[3],
            ray_origin=ray_origin,
            ray_direction=ray_direction,
        )
        # T.append(time.time() - tt)

        if is_intersect:
            # tt = time.time()
            # is_include = surface_inclusion_test(ray_direction=ray_direction, p=intersect_p, surface_points=surface_points)
            is_include = surface_inclusion_test(
                p=intersect_p,
                xy_bound_points=xy_bound_points,
                xz_bound_points=xz_bound_points,
                yz_bound_points=yz_bound_points,
            )
            # T.append(time.time() - tt)
            if is_include:
                normal = get_surface_normal(
                    A=0,
                    B=0,
                    C=0,
                    D=w_list[0],
                    E=w_list[1],
                    F=w_list[2],
                    point=intersect_p,
                )
                if np.dot(ray_direction, normal) > 0:
                    normal = -normal  # Reverse the normal
                intersect_t = np.concatenate(
                    [np.reshape(intersect_p, (1, -1)), np.reshape(normal, (1, -1))],
                    axis=0,
                )
                intersect_t_list.append(intersect_t)
    # T.append(time.time() - tt)

    if len(intersect_t_list) == 0:
        return False, None
    elif len(intersect_t_list) == 1:
        return True, intersect_t_list[0]
    else:
        closest_distance = 1e6
        closest_intersect_t = None
        for intersect_t in intersect_t_list:
            temp_d = np.linalg.norm(intersect_t[0] - ray_origin)
            if temp_d < closest_distance:
                closest_distance = temp_d
                closest_intersect_t = intersect_t
        return True, closest_intersect_t


def generate_half_sphere_points(center, radius, num_points, direction):
    """
    Generate points on the surface of a half-sphere in a specified direction.

    Parameters:
    - center: [c_x, c_y, c_z] coordinates of the sphere's center
    - radius: Radius of the sphere
    - num_points: Number of points to initially sample on the sphere's surface
    - direction: [d_x, d_y, d_z] a vector indicating the direction of the half-sphere to retain points

    Returns:
    - points: An array of points on the specified half-sphere's surface.
    """
    num_points = num_points * 2
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = np.arccos(
        1 - 2 * np.linspace(0.5, num_points - 0.5, num_points) / num_points
    )
    phi = golden_angle * np.arange(1, num_points + 1)

    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)

    points = np.column_stack((x, y, z))

    # Normalize the direction vector
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Calculate dot product between direction vector and points vectors
    dot_products = np.dot(points - center, direction)

    # Filter points based on the dot product (positive dot product means the point is in the specified half-sphere)
    half_sphere_points = points[dot_products > 0]

    return half_sphere_points


def ray_tracer(
    intersect_1_cnt: int,
    intersect_2_cnt: int,
    tx_loc: list,
    tx_ori: list,
    surface_list: list,
    rx_loc: list,
    r: float,
    max_order: int,
    phy_charac_dict: dict,
    omnidirectional: bool,
    angle=None,
    src_ori=None,
    traverseT: bool = True,
    first_p: list = None,
):
    """
    from tx, to rx sphere with radius r

    reverse相比forward的主要区别在于，如果 lis 和 first intersection之间有其他平面阻挡，那么这个path就作废
    """
    path_dict = {}
    distance = 0
    specular_attenuation = 1.0
    ray_origin = tx_loc
    ray_direction = tx_ori
    segment_list = []  # [max_order+2, 3]
    segment_list.append(np.array(ray_origin))

    log_list = [ray_direction]

    for n in range(max_order + 1):
        if n == 0:
            """the original ray won't intersect with rx sphere"""
            # intersect, intersect_t = check_scene_intersection_aabb_3d(ray_origin=ray_origin,
            #                                                           ray_direction=ray_direction,
            #                                                           plane_aabb_dict_list=plane_aabb_dict_list)
            intersect, intersect_t = intersection_test(
                surface_list=surface_list,
                ray_origin=ray_origin,
                ray_direction=ray_direction,
            )

            if not intersect:
                # for ii in range(max_order+1):
                #     segment_list.append(np.array([-1000., -1000., -1000.]))
                sl = len(segment_list)
                for ii in range(max_order + 2 - sl):
                    segment_list.append(np.array([-1000.0, -1000.0, -1000.0]))
                return (
                    False,
                    None,
                    segment_list,
                    intersect_1_cnt,
                    intersect_2_cnt,
                    log_list,
                )
            else:
                intersect_1_cnt += 1
                reflect_p = intersect_t[0]
                if traverseT:
                    if np.mean(abs(reflect_p - first_p)) > 1e-4:
                        # for ii in range(max_order + 1):
                        #     segment_list.append(np.array([-1000., -1000., -1000.]))
                        sl = len(segment_list)
                        for ii in range(max_order + 2 - sl):
                            segment_list.append(np.array([-1000.0, -1000.0, -1000.0]))
                        return (
                            False,
                            None,
                            segment_list,
                            intersect_1_cnt,
                            intersect_2_cnt,
                            log_list,
                        )

                t_normal = intersect_t[1]

                log_list.append(reflect_p)
                log_list.append(t_normal)

                distance += get_euclidean_distance(ray_origin, reflect_p)
                specular_attenuation *= phy_charac_dict["reflect_aRatio"] * (
                    1 - phy_charac_dict["scatter_aRatio"]
                )

                if np.mean(reflect_p - ray_origin) == 0:
                    raise ValueError("reflection point and ray origin are the same")

                ray_direction = (reflect_p - ray_origin) - 2 * t_normal * np.dot(
                    (reflect_p - ray_origin), t_normal
                )
                ray_origin = reflect_p

                segment_list.append(np.array(reflect_p))
        else:
            intersect, Q, M = check_ray_sphere_intersection(
                ray_origin=ray_origin,
                ray_direction=ray_direction,
                sphere_centroid=rx_loc,
                sphere_radius=r,
            )
            if intersect:
                log_list.append([999, 999, 999])

                valid = True
                if not omnidirectional:
                    if get_angle_between_vec(src_ori, -ray_direction) > angle:
                        valid = False

                if valid:
                    # for ii in range(max_order - n):
                    #     segment_list.append(np.array([-1000., -1000., -1000.]))
                    segment_list.append(np.array(rx_loc))
                    sl = len(segment_list)
                    for ii in range(max_order + 2 - sl):
                        segment_list.append(np.array([-1000.0, -1000.0, -1000.0]))
                    # segment_list.append(np.array(rx_loc))

                    distance += get_euclidean_distance(ray_origin, rx_loc)
                    # segment_list.append(np.array([ray_origin, M]))
                    path_dict["energy"] = (
                        specular_attenuation * get_distance_attenuation(distance)
                    )
                    path_dict["delay"] = (
                        distance - get_euclidean_distance(tx_loc, rx_loc)
                    ) / phy_charac_dict["wave_speed"]
                    return (
                        True,
                        path_dict,
                        segment_list,
                        intersect_1_cnt,
                        intersect_2_cnt,
                        log_list,
                    )
                else:
                    # segment_list.append(np.array([-1000., -1000., -1000.]))
                    sl = len(segment_list)
                    for ii in range(max_order + 2 - sl):
                        segment_list.append(np.array([-1000.0, -1000.0, -1000.0]))
                    return (
                        False,
                        None,
                        segment_list,
                        intersect_1_cnt,
                        intersect_2_cnt,
                        log_list,
                    )
            else:
                if n != max_order:
                    intersect_2, intersect_t = intersection_test(
                        surface_list=surface_list,
                        ray_origin=ray_origin,
                        ray_direction=ray_direction,
                    )

                    if not intersect_2:
                        # for ii in range(max_order + 1 - n):
                        #     segment_list.append(np.array([-1000., -1000., -1000.]))
                        sl = len(segment_list)
                        for ii in range(max_order + 2 - sl):
                            segment_list.append(np.array([-1000.0, -1000.0, -1000.0]))
                        return (
                            False,
                            None,
                            segment_list,
                            intersect_1_cnt,
                            intersect_2_cnt,
                            log_list,
                        )
                    else:
                        intersect_2_cnt += 1
                        reflect_p = intersect_t[0]
                        t_normal = intersect_t[1]

                        distance += get_euclidean_distance(ray_origin, reflect_p)
                        specular_attenuation *= phy_charac_dict["reflect_aRatio"] * (
                            1 - phy_charac_dict["scatter_aRatio"]
                        )

                        if np.mean(reflect_p - ray_origin) == 0:
                            raise ValueError(
                                "reflection point and ray origin are the same"
                            )

                        ray_direction = (
                            reflect_p - ray_origin
                        ) - 2 * t_normal * np.dot((reflect_p - ray_origin), t_normal)
                        ray_origin = reflect_p

                        segment_list.append(np.array(reflect_p))

                else:
                    sl = len(segment_list)
                    for ii in range(max_order + 2 - sl):
                        segment_list.append(np.array([-1000.0, -1000.0, -1000.0]))
                    return (
                        False,
                        None,
                        segment_list,
                        intersect_1_cnt,
                        intersect_2_cnt,
                        log_list,
                    )

    raise RuntimeError("ray_tracer_directional work error ")


def ray_tracing(
    surface_list: list,
    ray_num,
    src_loc,
    src_ori,
    detector_r,
    lis_loc,
    phy_charac_dict,
    max_order,
    ir_len,
    fs,
):
    """
    1, generate rays in defined azimuth/elevation range according to a certain resolution
    2, for each ray, perform ray-tracing iteratively, stop ray-tracing when:
        - there's no intersection with triangle or lis_detector
        - intersect with lis_detector
    3, after all rays are finished, calculate ir according to recorded paths


    """

    ray_ori_list = generate_half_sphere_points(
        center=np.array([0, 0, 0]), radius=1, num_points=ray_num, direction=src_ori
    )

    direct_energy = get_distance_attenuation(get_euclidean_distance(src_loc, lis_loc))

    path_dict_list = []
    # ray_num = 0
    # staticNum = 1000
    # t1 = time.time()
    """
    segment_1: transmitter - reflect 1
    segment_2: reflect 1 - reflect 2
    segment_3: reflect 2 - receiver
    """
    overall_segment_list = []  # [N, max_order+2, 3]
    print("search for %d rays" % len(ray_ori_list))
    intersect_1_cnt = 0
    intersect_2_cnt = 0
    total_log_list = []

    times = []

    for ray_direction in ray_ori_list:
        # ray_num += 1
        # if ray_num % staticNum == 0:
        #     print("ray tracing, %d/%d, consumes %fs" % (ray_num, ray_ori_list.shape[0], time.time()-t1))
        #     t1 = time.time()
        ray_origin = src_loc

        t = time.time()
        (
            path_exist,
            path_dict,
            segment_list,
            intersect_1_cnt,
            intersect_2_cnt,
            log_list,
        ) = ray_tracer(
            intersect_1_cnt=intersect_1_cnt,
            intersect_2_cnt=intersect_2_cnt,
            tx_loc=ray_origin,
            tx_ori=ray_direction,
            surface_list=surface_list,
            rx_loc=lis_loc,
            r=detector_r,
            max_order=max_order,
            phy_charac_dict=phy_charac_dict,
            omnidirectional=True,
            traverseT=False,
        )

        times.append(time.time() - t)

        overall_segment_list.append(segment_list)
        if path_exist:
            # print(segment_list)
            path_dict_list.append(path_dict)

        if len(log_list) > 1:
            if len(log_list) == 4:
                log_string = f"orientation: {log_list[0]}, reflet_p: {log_list[1]}, reflect_normal: {log_list[2]}, found"
            else:
                log_string = f"orientation: {log_list[0]}, reflet_p: {log_list[1]}, reflect_normal: {log_list[2]}"
            total_log_list.append(log_string)

    if np.array(overall_segment_list).shape[1] != max_order + 2:
        raise ValueError("segment_list reserve error")

    times = np.array(times)

    print(f"intersect_1_cnt: {intersect_1_cnt}")
    print(f"intersect_2_cnt: {intersect_2_cnt}")

    ir = np.zeros(shape=(int(ir_len * fs),))

    print("find %d valid paths" % len(path_dict_list))

    peaks = []
    if len(path_dict_list) == 0:
        ir[0] = 1.0
        print(f"peaks found: {peaks}")
        return ir, None, np.array(overall_segment_list), times, total_log_list
    else:
        ir[0] = direct_energy
        for path_dict in path_dict_list:
            peak_idx = int(path_dict["delay"] * fs)
            if peak_idx < np.floor(ir_len * fs):
                if ir[peak_idx] == 0:
                    ir[peak_idx] = path_dict["energy"]
                else:
                    ir[peak_idx] = (ir[peak_idx] + path_dict["energy"]) / 2
            if ir[peak_idx] < 0.1 * direct_energy:
                ir[peak_idx] += 0.1 * direct_energy
            peaks.append(peak_idx)
        ir = ir / max(abs(ir))
        print(f"peaks found: {peaks}")
        return (
            ir,
            np.array(path_dict_list),
            np.array(overall_segment_list),
            times,
            total_log_list,
        )


def extract_boundary_points(points, epsilon=0.01):
    """
    Extracts the boundary points of a set of 2D points using the convex hull.

    Parameters:
    - points: A (N, 2) numpy array of points in 2D space.

    Returns:
    - A numpy array of boundary points of the convex hull.
    """
    hull = ConvexHull(points)
    boundary_points = points[hull.vertices]
    # return boundary_points
    # Create a polygon from the boundary points
    polygon = Polygon(boundary_points)

    # Simplify the polygon's boundary using the Ramer-Douglas-Peucker algorithm
    simplified_polygon = polygon.simplify(epsilon, preserve_topology=False)

    # Extract the coordinates of the simplified boundary
    simplified_points = np.array(simplified_polygon.exterior.coords)

    return simplified_points


def _calc_angle(frame, offset, HFOV):
    return math.atan(math.tan(HFOV / 2.0) * offset / (frame.shape[1] / 2.0))


def calc_spatials(depthFrame, roi, FOV, rm_ratio, averaging_method=np.mean):
    # calibData = device.readCalibration()
    assert rm_ratio * 2 < 1

    # Values
    DELTA = 5
    THRESH_LOW = 200  # 20cm
    THRESH_HIGH = 10000  # 10m

    # roi = _check_input(roi, depthFrame)  # If point was passed, convert it to ROI
    xmin, ymin, xmax, ymax = roi
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)

    # Calculate the average depth in the ROI.
    # depthROI = depthFrame[ymin:ymax, xmin:xmax]
    depthROI = depthFrame[ymin:ymax, xmin:xmax]
    inRange = (THRESH_LOW <= depthROI) & (depthROI <= THRESH_HIGH)

    # Required information for calculating spatial coordinates on the host
    # HFOV = np.deg2rad(calibData.getFov(dai.CameraBoardSocket(depthData.getInstanceNum())))
    # HFOV = np.deg2rad(calibData.getFov(dai.CameraBoardSocket.CAM_C))
    HFOV = np.deg2rad(FOV)

    # averageDepth = averaging_method(depthROI[inRange])

    filtered_depthROI = depthROI[inRange]
    filtered_depthROI = np.reshape(filtered_depthROI, (-1,))
    sorted_depthROI = np.sort(filtered_depthROI)
    rm_num = int(len(sorted_depthROI) * rm_ratio)
    final_depthROI = sorted_depthROI[rm_num:-rm_num]

    # depth_list = np.abs(final_depthROI)

    averageDepth = averaging_method(final_depthROI)

    centroid = {  # Get centroid of the ROI
        "x": int((xmax + xmin) / 2),
        "y": int((ymax + ymin) / 2),
    }

    midW = int(depthFrame.shape[1] / 2)  # middle of the depth img width
    midH = int(depthFrame.shape[0] / 2)  # middle of the depth img height
    bb_x_pos = centroid["x"] - midW
    bb_y_pos = centroid["y"] - midH

    angle_x = _calc_angle(depthFrame, bb_x_pos, HFOV)
    angle_y = _calc_angle(depthFrame, bb_y_pos, HFOV)

    spatials = {
        "z": averageDepth,
        "x": averageDepth * math.tan(angle_x),
        "y": -averageDepth * math.tan(angle_y),
    }
    return spatials, centroid


def reflector_selection_20240115(ann, rN):
    pixel_area_list = np.sum(ann, axis=(1, 2))
    top_idx = np.argsort(pixel_area_list)[::-1]
    return ann[top_idx[rN[0] : rN[1]]]


def extract_values_and_coordinates(A, B):
    """
    Extract values from B where the corresponding index in A is 1,
    along with their image coordinates.

    Args:
    - A (np.array): A binary mask array with shape [h, w].
    - B (np.array): A depth map array with shape [h, w].

    Returns:
    - extracted_values (np.array): 1D array of extracted values from B.
    - coordinates (np.array): 2D array of shape [N, 2], representing image coordinates (u, v).
    """
    # Create a mask where A == 1
    mask = A == 1

    # Extract values from B using the mask
    extracted_values = B[mask]

    # Find the indices in A where A == 1, which are the coordinates
    v, u = np.where(mask)

    # Stack the coordinates in a 2D array [N, 2]
    # coordinates = np.stack((u, v), axis=-1)

    return extracted_values, u, v


from scipy.optimize import fsolve


def solve_for_z(x, y, D, E, F, G):
    # This function assumes C != 0 and rearranges the equation to z = f(x, y)
    func = lambda z: D * x + E * y + F * z + G
    z_initial_guess = 0
    (z_solution,) = fsolve(func, z_initial_guess)
    return z_solution


def solve_for_x(y, z, D, E, F, G):
    # This function assumes C != 0 and rearranges the equation to z = f(x, y)
    func = lambda x: D * x + E * y + F * z + G
    x_initial_guess = 0
    (x_solution,) = fsolve(func, x_initial_guess)
    return x_solution


def solve_for_y(x, z, D, E, F, G):
    # This function assumes C != 0 and rearranges the equation to z = f(x, y)
    func = lambda y: D * x + E * y + F * z + G
    y_initial_guess = 0
    (y_solution,) = fsolve(func, y_initial_guess)
    return y_solution


def sample_points(
    x_min, x_max, y_min, y_max, z_min, z_max, D, E, F, G, density, normal
):
    x_range = np.arange(x_min, x_max, density)
    y_range = np.arange(y_min, y_max, density)
    z_range = np.arange(z_min, z_max, density)

    x_normal = np.array([1, 0, 0])
    y_normal = np.array([0, 1, 0])
    z_normal = np.array([0, 0, 1])

    points = []
    if np.all(normal == x_normal):
        for y in y_range:
            for z in z_range:
                x = solve_for_x(y, z, D, E, F, G)
                if x_min <= x <= x_max:
                    points.append(np.array([x, y, z]))
    elif np.all(normal == y_normal):
        for x in x_range:
            for z in z_range:
                y = solve_for_y(x, z, D, E, F, G)
                if y_min <= y <= y_max:
                    points.append(np.array([x, y, z]))
    elif np.all(normal == z_normal):
        for x in x_range:
            for y in y_range:
                z = solve_for_z(x, y, D, E, F, G)
                if z_min <= z <= z_max:
                    points.append(np.array([x, y, z]))
    else:
        raise ValueError("sampling surface error, input normal wrong")

    return points


def get_closest_plane_normal(direction_vector):
    """
    Finds the axis plane (XY, XZ, YZ) that has the normal vector with the smallest angle
    to the given direction vector and returns the normal of that plane.

    Parameters:
    - direction_vector: A numpy array representing the direction vector.

    Returns:
    - The normal vector of the plane (XY, XZ, or YZ) that has the smallest angle
      to the direction vector.
    """
    # Normalize the direction vector to ensure a fair comparison
    norm_dir_vec = direction_vector / np.linalg.norm(direction_vector)

    # Abs value of components for comparison
    abs_components = np.abs(norm_dir_vec)

    # Determine which component is smallest since its corresponding plane
    # will have the normal vector making the smallest angle with the direction vector
    # min_index = np.argmin(abs_components)
    max_index = np.argmax(abs_components)

    # Normals to the planes: XY, XZ, YZ
    # normals = [np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0])]
    normals = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

    # Return the normal of the closest plane
    return normals[max_index]


import json


def extract_bounding_boxes(json_data):
    """
    Extracts bounding box coordinates from LabelMe JSON format.

    Parameters:
    - json_data: dict, loaded JSON data from LabelMe annotation.

    Returns:
    - A list of tuples, where each tuple contains the bounding box coordinates
      (x1, y1, x2, y2) for a labeled object.
    """
    bounding_boxes = []

    # Check if 'shapes' key is in the JSON data
    if "shapes" in json_data:
        for shape in json_data["shapes"]:
            if shape["shape_type"] == "rectangle":
                # Assuming the points are top-left and bottom-right corners
                points = shape["points"]
                x1, y1 = points[0]
                x2, y2 = points[1]
                bounding_boxes.append((x1, y1, x2, y2))

    return bounding_boxes


def apply_masks_on_image(image, masks, colors, border_thickness):
    # Ensure image has an alpha channel
    if image.shape[2] == 3:
        image = np.concatenate(
            [image, np.ones((image.shape[0], image.shape[1], 1)) * 255], axis=-1
        ).astype(np.float32)

    structure = np.ones((3, 3))

    for i, mask in enumerate(masks):
        alpha = colors[i][3] / 255.0  # Normalize alpha to [0, 1]

        # Find mask border by erosion and subtraction
        eroded_mask = binary_erosion(
            mask, structure=iterate_structure(structure, border_thickness)
        ).astype(mask.dtype)
        border_mask = mask - eroded_mask

        for j in range(3):  # Apply color to RGB channels
            overlay_color = colors[i][j]
            # Apply color where mask is 1 but not on the border
            image[:, :, j] = np.where(
                (mask == 1) & (border_mask == 0),
                overlay_color * alpha + image[:, :, j] * (1 - alpha),
                image[:, :, j],
            )
            # Apply black color on border
            image[:, :, j] = np.where(border_mask == 1, 0, image[:, :, j])

        # Update alpha channel for mask area considering existing alpha
        existing_alpha = image[:, :, 3] / 255.0  # Normalize existing alpha to [0, 1]
        # Adjust alpha channel based on mask and new alpha value, including border
        image[:, :, 3] = np.where(
            mask == 1, 255 * np.maximum(alpha, existing_alpha), image[:, :, 3]
        )

    return image.astype(np.uint8)


def apply_mask_and_blur(image_path, mask, blur):
    # 加载图像
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    # 将遮罩矩阵转换为PIL图像
    mask_image = Image.fromarray(np.uint8(mask * 255), "L")

    # 应用模糊效果
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur))

    # 创建一个灰色蒙版
    grey_overlay = Image.new("RGBA", image.size, (128, 128, 128, 128))  # 灰色蒙版
    blurred_with_overlay = Image.alpha_composite(blurred_image, grey_overlay)

    # 使用遮罩合并原图和处理后的图像
    final_image = Image.composite(blurred_with_overlay, image, mask_image)

    return final_image


def get_simu_peaks(array, zero_tolerance=1):
    max_indices = []
    current_cluster_max = None
    current_cluster_max_index = None
    zero_count = 0  # Count of consecutive zeros within a potential cluster

    for i, value in enumerate(array):
        if value != 0:
            # Reset zero count when a non-zero is found
            zero_count = 0
            # Update current cluster max and index if this value is larger, or if it's the first non-zero value
            if current_cluster_max is None or value > current_cluster_max:
                current_cluster_max = value
                current_cluster_max_index = i
        else:
            # Increment zero count and check against the tolerance
            zero_count += 1
            if zero_count > zero_tolerance:
                # When exceeding tolerance, if a cluster has ended, save its max index
                if current_cluster_max is not None:
                    max_indices.append(current_cluster_max_index)
                    current_cluster_max = None  # Reset for the next cluster
                zero_count = 1  # Reset zero count for any potential new cluster

    # Check if the last element is part of a cluster
    if current_cluster_max is not None:
        max_indices.append(current_cluster_max_index)

    return np.array(max_indices)


def png2mask(png):
    is_red = (png[:, :, 0] == 0) & (png[:, :, 1] == 0) & (png[:, :, 2] == 0)
    mask = is_red.astype(int)
    mask = 1 - mask
    mask = mask[np.newaxis, :, :]
    return mask


import random


def randomly_select_ones(matrix, W):
    # Get the shape of the matrix
    M, N = matrix.shape

    # Find the indices where elements are 1
    ones_indices = np.argwhere(matrix == 1)

    # Check if we have enough ones to select W elements
    if len(ones_indices) < W:
        raise ValueError(
            "There are fewer ones in the matrix than the number of selections requested"
        )

    # Randomly choose W indices from the ones_indices
    selected_indices = random.sample(list(ones_indices), W)

    # Create a new matrix of zeros
    new_matrix = np.zeros((M, N), dtype=int)

    # Set only the selected positions to 1
    for idx in selected_indices:
        new_matrix[tuple(idx)] = 1

    return new_matrix


def calculate_distances(points, ref_point):
    # Calculate differences along each dimension
    differences = points - ref_point

    # Calculate squared differences
    squared_differences = differences**2

    # Sum the squared differences across columns (i.e., for each point)
    sum_of_squares = np.sum(squared_differences, axis=1)

    # Take the square root to get the Euclidean distances
    distances = np.sqrt(sum_of_squares)

    return distances


def get_lis_reflector_distance(mask, depth, intrinsics, samp_num, rm_ratio, lis_loc):
    sparse_mask = randomly_select_ones(mask, samp_num)
    fx, fy, cx, cy = (
        intrinsics[0][0],
        intrinsics[1][1],
        intrinsics[0][2],
        intrinsics[1][2],
    )
    R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)

    filtered_depth_list, u_list, v_list = extract_values_and_coordinates(
        A=sparse_mask, B=depth
    )

    z_list = filtered_depth_list / 1000
    x_list = (u_list - cx) * z_list / fx
    y_list = (v_list - cy) * z_list / fy

    points = np.stack((x_list, y_list, z_list), axis=-1)
    points = np.dot(points, R_camera_to_world)
    dist_list = calculate_distances(points, lis_loc)
    dist_list = np.sort(dist_list)
    half_rm_num = int(samp_num * rm_ratio)
    dist_list = dist_list[half_rm_num:-half_rm_num]
    avg_dist = np.mean(dist_list)

    return avg_dist


def reflector_lis_dist_ranking(
    original_mask_list, depth, intrinsics, samp_num, rm_ratio, lis_loc
):
    avg_dist_list = []
    for m in original_mask_list:
        avg_dist_list.append(
            get_lis_reflector_distance(
                mask=m,
                depth=depth,
                intrinsics=intrinsics,
                samp_num=samp_num,
                rm_ratio=rm_ratio,
                lis_loc=lis_loc,
            )
        )
    avg_dist_list = np.array(avg_dist_list)
    dist_idx_rank = np.argsort(avg_dist_list)
    """越小越好"""
    return original_mask_list[dist_idx_rank]


import struct
import os
import scipy.io as sio


def read_vector_from_dat_file(filename):
    with open(filename, "rb") as file:
        # Read vector size
        # size_bytes = file.read(8)  # Assuming size_t is 8 bytes
        # size = struct.unpack("Q", size_bytes)[0]
        size = 3
        # Read vector elements
        data_bytes = file.read(size * 8)  # Assuming double is 8 bytes
        data = struct.unpack("d" * size, data_bytes)

        return list(data)


def read_surface_list(surf_path):
    try:
        with open(surf_path, "rb") as file:
            surfaceList = []
            while True:
                A_data = file.read(32)  # Assuming double is 8 bytes, 4 doubles
                if not A_data:
                    break
                A = struct.unpack("4d", A_data)

                B, C, D = [], [], []
                for vec in (B, C, D):
                    rows_cols = file.read(8)  # Reading two integers (rows and cols)
                    if not rows_cols:
                        break
                    rows, cols = struct.unpack("2i", rows_cols)
                    if cols != 2:
                        raise ValueError("Unexpected column size, must be 2.")

                    for _ in range(rows):
                        data = file.read(16)  # Reading 2 doubles per row
                        if not data:
                            break
                        vec.append(struct.unpack("2d", data))

                surfaceList.append((A, B, C, D))

            print("surfaceList size:", len(surfaceList))
            return surfaceList
    except IOError:
        print("Cannot open the file!")
        return None
    except struct.error:
        print("Error processing binary data.")
        return None


def simple_trace(
    surface_info: list, lis_coord: np.ndarray, ray_num=200000, sample_freq=48000
):
    t1 = time.time()
    src_loc = np.array([0, 0, 0])
    src_ori = np.array([0, 0, 1])
    max_reflect_order = 1  # 最多看一次反射的光线
    phy_charac_dict = {"wave_speed": 343, "reflect_aRatio": 0.5, "scatter_aRatio": 0.0}
    detector_r = 0.05
    ir_T = 0.04
    LOS_T = 0.0005
    ir, path_dict_list, overall_segment_list, times, log_list = ray_tracing(
        surface_list=surface_info,
        ray_num=ray_num,
        src_loc=src_loc,
        src_ori=src_ori,
        detector_r=detector_r,
        lis_loc=lis_coord,
        phy_charac_dict=phy_charac_dict,
        max_order=max_reflect_order,
        ir_len=ir_T,
        fs=sample_freq,
    )

    simu_ir = np.copy(ir)
    simu_ir[: int(sample_freq * LOS_T)] = 0
    simu_peaks = get_simu_peaks(array=simu_ir, zero_tolerance=10)

    runT = (time.time() - t1) * 1000
    print(f"ray tracing, ray num {ray_num}, run time: {runT} ms.")
    return simu_peaks


if __name__ == "__main__":
    ray_num_list = [200000]
    reflector_num_list = [3]

    fs = 96000
    src_loc = np.array([0, -0.14, 0])
    detector_r = 0.05
    src_ori = np.array([0, 0, -1])
    max_reflect_order = 1
    ir_T = 0.04
    phy_charac_dict = {"wave_speed": 343, "reflect_aRatio": 0.8, "scatter_aRatio": 0.2}
    LOS_T = 0.0005

    test_num = 100
    """ ray tracing  """

    for reflector_num in reflector_num_list:
        for ray_num in ray_num_list:
            stat_list = []
            simu_peak_ls = []
            for ii in range(test_num):
                surf_path = f"/home/seldon/Downloads/code_data_toZZX/test_data/surface_{reflector_num}_{ii}.dat"  # surface_XXX.dat
                lis_path = f"/home/seldon/Downloads/code_data_toZZX/test_data/rxLoc_{reflector_num}_{ii}.dat"  # rxLoc_XXX.dat
                surface_list = read_surface_list(surf_path)
                print(
                    "surface list: ",
                    surface_list[0][0],
                    "\n",
                    surface_list[0][1],
                    type(surface_list),
                )
                lis_loc = read_vector_from_dat_file(lis_path)
                lis_loc = np.array(lis_loc)

                t1 = time.time()
                ir, path_dict_list, overall_segment_list, times, log_list = ray_tracing(
                    surface_list=surface_list,
                    ray_num=ray_num,
                    src_loc=src_loc,
                    src_ori=src_ori,
                    detector_r=detector_r,
                    lis_loc=lis_loc,
                    phy_charac_dict=phy_charac_dict,
                    max_order=max_reflect_order,
                    ir_len=ir_T,
                    fs=fs,
                )
                # print(f"ray tracing, run time: %f" % (time.time() - t1))

                simu_ir = np.copy(ir)
                simu_ir[: int(fs * LOS_T)] = 0
                simu_peaks = get_simu_peaks(array=simu_ir, zero_tolerance=10)

                runT = (time.time() - t1) * 1000
                stat_list.append(runT)
                print(f"ray tracing, {reflector_num}, {ray_num}, run time: {runT}")
                if simu_peaks is not None:
                    simu_peak_ls.append(simu_peaks)

            stat_list = np.array(stat_list)
            save_path = f"/home/seldon/Downloads/code_data_toZZX/result/delay_python_{reflector_num}_{ray_num}.mat"
            sio.savemat(save_path, mdict={"data": stat_list})
            print(simu_peak_ls)
