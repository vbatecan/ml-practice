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
