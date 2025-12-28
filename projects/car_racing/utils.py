import math
import random
import pygame

def lerp(a, b, t):
    return a + (b - a) * t

def get_spline_point(p0, p1, p2, p3, t):
    """
    Catmull-Rom spline interpolation.
    Returns a point at t [0, 1] between p1 and p2.
    """
    t2 = t * t
    t3 = t2 * t
    
    # Coefficients for the cubic polynomial
    q0 = -t3 + 2*t2 - t
    q1 = 3*t3 - 5*t2 + 2
    q2 = -3*t3 + 4*t2 + t
    q3 = t3 - t2
    
    tx = 0.5 * (p0[0] * q0 + p1[0] * q1 + p2[0] * q2 + p3[0] * q3)
    ty = 0.5 * (p0[1] * q0 + p1[1] * q1 + p2[1] * q2 + p3[1] * q3)
    
    return (tx, ty)

def generate_track_points(center, num_points, min_radius, max_radius):
    """
    Generates random control points around a center.
    """
    points = []
    for i in range(num_points):
        angle = (2 * math.pi / num_points) * i
        # Add some random variation to the angle for irregularity
        angle += random.uniform(-0.1, 0.1)
        radius = random.uniform(min_radius, max_radius)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points

def compute_track_polygon(control_points, resolution=20):
    """
    Interpolates control points to create a smooth closed loop polygon.
    """
    smooth_points = []
    count = len(control_points)
    
    for i in range(count):
        # 4 control points for Catmull-Rom: p0, p1, p2, p3
        # We interpolate between p1 and p2
        p0 = control_points[(i - 1) % count]
        p1 = control_points[i]
        p2 = control_points[(i + 1) % count]
        p3 = control_points[(i + 2) % count]
        
        for j in range(resolution):
            t = j / resolution
            pos = get_spline_point(p0, p1, p2, p3, t)
            smooth_points.append(pos)
            
    return smooth_points

def get_closest_point_on_path(pos, path):
    """
    Finds the index of the closest point in the path to the given position.
    """
    min_dist_sq = float('inf')
    closest_idx = -1
    
    # Optimization: If we knew the previous index, we could search locally.
    # But linear search is fast enough for < vài thousand points.
    px, py = pos[0], pos[1]
    
    for i, (tx, ty) in enumerate(path):
        dx = px - tx
        dy = py - ty
        dist_sq = dx*dx + dy*dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_idx = i
            
    return closest_idx, min_dist_sq**0.5

def calculate_cross_track_error(car_pos, path, closest_idx):
    """
    Calculates signed distance from track center.
    Negative = Left of track, Positive = Right of track (relative to track direction).
    """
    if closest_idx < 0 or closest_idx >= len(path): return 0
    
    # Track point
    p_current = pygame.math.Vector2(path[closest_idx])
    
    # Next point (looping)
    next_idx = (closest_idx + 1) % len(path)
    p_next = pygame.math.Vector2(path[next_idx])
    
    # Track direction vector
    track_dir = p_next - p_current
    if track_dir.length() == 0: return 0
    track_dir = track_dir.normalize()
    
    # Vector from track point to car
    to_car = car_pos - p_current
    
    # Cross product 2D (z-component)
    # CP = ax*by - ay*bx
    cross_product = track_dir.x * to_car.y - track_dir.y * to_car.x
    
    # Distance is approximately the length of the projection on normal, 
    # but since we found the closest point, |to_car| is rough dist.
    # More accurate: perpendicular distance to the line segment.
    # But simple distance * sign is usually sufficient for RL if points are dense.
    
    dist = to_car.length()
    
    # Sign: If cross_product > 0, car is "right" or "left" depending on coord system.
    # Pygame Y is down. 
    # Let's say Track is East (1,0). Car is South (0,1). CP = 1*1 - 0 = 1.
    # In screen space, South is "Right" if moving East? No, South is Down.
    # Standard: Left = positive CTE, Right = negative CTE? Or vice versa.
    # We'll just return signed distance.
    return dist if cross_product > 0 else -dist

def calculate_heading_error(car_angle, car_pos, path, closest_idx, lookahead=10):
    """
    Calculates the angular difference between car heading and target track point.
    Returns value in degrees [-180, 180].
    """
    if closest_idx < 0 or closest_idx >= len(path): return 0

    # Target point
    target_idx = (closest_idx + lookahead) % len(path)
    target_pos = pygame.math.Vector2(path[target_idx])
    
    # Vector to target
    to_target = target_pos - car_pos
    if to_target.length() == 0: return 0
    
    # Angle of vector to target
    # atan2(y, x) -> radians. -y because pygame Y is flipped? 
    # Pygame Vector2.angle_to() is convenient relative to (1,0).
    # car.angle is in degrees, 0 = Up (usually) or Right depending on initiation.
    # In Car.__init__: 
    # heading = (-sin(rad), -cos(rad)). 0 deg => (0, -1) UP.
    
    # Let's calculate target angle in degrees in standard math (0=Right, CCW+)
    # Then convert to Pygame Car convention (0=Up, CW+? or whatever it is).
    
    # Actually, easiest way: 
    # Get Car Forward Vector
    rad = math.radians(car_angle)
    car_forward = pygame.math.Vector2(-math.sin(rad), -math.cos(rad))
    
    # Get Angle between Car Forward and To Target
    angle_diff = car_forward.angle_to(to_target) 
    
    # Vector2.angle_to returns angle from self to other.
    # It accounts for coordinate system.
    
    # Ensure wrap around -180 to 180
    while angle_diff > 180: angle_diff -= 360
    while angle_diff < -180: angle_diff += 360
    
    return -angle_diff # Negate if necessary to match steering sign conventions (Left+, Right-)
