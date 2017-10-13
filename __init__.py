# coding=utf-8

"""
"""

__author__ = "Morten Lind"
__copyright__ = " 2017"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten.lind@sintef.no"
__status__ = "Development"

import numpy as np
import math3d as m3d


def crop_roi(npc, roi):
    """Given a npy-point cloud and an 'roi' in the same coordinate
    reference, return the cropped point cloud.
    """
    return npc[(npc[:, 0] > roi[0][0]) & (npc[:, 0] < roi[0][1]) &
               (npc[:, 1] > roi[1][0]) & (npc[:, 1] < roi[1][1]) &
               (npc[:, 2] > roi[2][0]) & (npc[:, 2] < roi[2][1])]


def projection_along(npc, direction):
    """Calculate the projection of the cloud points along the given
    'direction'. If 'direction' is a unit vector, measurement units will
    be preserved."""
    if type(direction) == m3d.Vector:
        direction = direction.array
    return npc.dot(direction)


def crop_along(npc, origo, direction, limits):
    """Crop the point cloud according to the 'limits', which is a pair,
    when projected on 'direction' and measured from position 'origo'.
    """
    if type(origo) == m3d.Vector:
        origo = origo.array
    if type(direction) == m3d.Vector:
        direction = direction.array
    proj = (npc - origo).dot(direction)
    return npc[(proj > limits[0]) & (proj < limits[1])]


def transform_roi(b_in_a, roi_b):
    """Given the pose of reference 'b' in reference 'a', 'b_in_a' and a
    ROI with reference in 'b', calculate the approximate ROI in 'a'
    reference, assuming that axes among 'a' and 'b' are approximately
    aligned, i.e. the rotation part of the 'b_in_a' transform is
    nearly a permutation of axes.
    """
    roi_a = np.empty((3,2))
    axis_match_threshold = 0.8
    bo = b_in_a.orient
    bp = b_in_a.pos
    ao = np.identity(3)
    for i in range(3):
        match_j = None
        for j in range(3):
            if np.abs(ao[i].dot(bo[j])) > axis_match_threshold:
                match_j = j
                break
        if match_j is None:
            raise Exception('pallet_finder: Unable to find matching axis for direction {}'.format(i))
        # Fix cut off in base axis i by projection from pallet origo
        # along pallet direction j
        print('Associating direction {} of the pallet with {} of the base'.format(j,i))
        roi_a[i] = sorted([ao[i].dot(bp.array + roi_b[j][0]*bo[j]),
                           ao[i].dot(bp.array + roi_b[j][1]*bo[j])
        ])
    return roi_a
