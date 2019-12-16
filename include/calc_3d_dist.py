#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import numpy as np


def point_plane_distance(points, plane_point, plane_normal):
    '''
    Arguments:
        points {np.ndarray}: (N, 3), [(x1, y1, z1), (x2, y2, z2), ...]
        plane_point {np.ndarray}: (3, ), (x0, y0, z0)
        plane_normal {np.ndarray}: (3, ), (A, B, C)
    Return:
        dists {np.ndarray}: (N, )
            Positive or negative.
    '''
    # Distance between point and plane.
    # https://mathinsight.org/distance_point_plane
    # plane normal vector: (A, B, C)
    # point Q=(x0, y0, z0)
    # plane: A(x−x0)+B(y−y0)+C(z−z0)=0
    # plane: Ax+By+Cx+D=0
    #   D=-Ax0-By0-Cz0
    D = -plane_normal.dot(plane_point)
    numerator = points.dot(plane_normal) + D  # (N, 1)
    denominator = np.linalg.norm(plane_normal)  # (1, )
    dists = numerator / denominator
    return dists


def point_3d_line_distance(points, p1, p2):
    # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    '''
    Arguments:
        p1: a point on the 3d line.
        p2: a point on the 3d line.
        points {np.ndarray}: (N,3 )
            the points that we want to compute the distance to the 3d line.
    Return:
        dists {np.ndarray}: (N, )
            Positive. No negative.
    '''
    crs = np.cross((points - p1), (points - p2))
    dists = np.linalg.norm(crs, axis=1)  # (N, 1)
    dists /= np.linalg.norm(p2 - p1)
    return dists


if __name__ == '__main__':

    # Define points
    pts = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 2],
    ])
    
    # Define a plane and a line.
    # A plane formed by y-z axis,
    # A line of x axis.
    p1 = np.array([0., 0., 0.])
    p2 = np.array([1., 0., 0.])
    pn = p2 - p1

    # Compute distance
    dists_plane = point_plane_distance(pts, p1, p2-p1)
    dists_3d_line = point_3d_line_distance(pts, p1, p2)

    print("dists_plane: " + str(dists_plane))
    print("dists_3d_line: " + str(dists_3d_line))