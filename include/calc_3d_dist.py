#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Two functions:
def point_plane_distance
def point_3d_line_distance
'''

import math
import numpy as np


def point_plane_distance(points, plane_point, plane_normal):
    ''' Calculate Distance between point and plane.
    Arguments:
        points {np.ndarray}: (N, 3), [[x1, y1, z1], [x2, y2, z2], ...]
        plane_point {np.ndarray}: (3, ), (x0, y0, z0)
        plane_normal {np.ndarray}: (3, ), (A, B, C)
    Return:
        dists {np.ndarray}: (N, )
            The distance might be positive or negative.
            It's positive if the point is on the side of the plane normal.
    '''
    # https://mathinsight.org/distance_point_plane
    # plane normal vector: (A, B, C)
    # plane point Q=(x0, y0, z0)
    # plane eq: A(x−x0)+B(y−y0)+C(z−z0)=0
    # plane eq: Ax+By+Cx+D=0, where D=-Ax0-By0-Cz0
    D = -plane_normal.dot(plane_point)
    numerator = points.dot(plane_normal) + D  # (N, 1)
    denominator = np.linalg.norm(plane_normal)  # (1, )
    dists = numerator / denominator
    return dists


def point_3d_line_distance(points, p1_on_3d_line, p2_on_3d_line):
    ''' Calculate the distance between 3D line and each point. 
    Arguments:
        p1_on_3d_line {np.ndarray}: shape=(3, ).
        p2_on_3d_line {np.ndarray}: shape=(3, ).
        points {np.ndarray}: shape=(N, 3).
            the points that we want to compute the distance to the 3d line.
    Return:
        dists {np.ndarray}: shape=(N, ).
            All distances are non-negative.
    '''
    # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    crs = np.cross((points - p1_on_3d_line), (points - p2_on_3d_line))
    dists = np.linalg.norm(crs, axis=1)  # (N, )
    dists /= np.linalg.norm(p2_on_3d_line - p1_on_3d_line)
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
    assert(np.allclose(dists_plane, [0., 1., 0.]))
    assert(np.allclose(dists_3d_line, [0., 1.41421356, 2.]))
    print("All test cases assert True!")