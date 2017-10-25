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
import pcl


def transform_npc(npc, trf):
    """Apply the transform 'trf' to the numpy point cloud 'npc', returning
    the transformed point cloud.
    """
    return trf.orient.array.dot(npc.T).T + trf.pos.array


def transform_pc(pc, trf):
    """Apply the transform 'trf' to the pcl.PointCloud object 'pc',
    returning the transformed point cloud with correctly transformed
    sensor pose.
    """
    new_pc = pcl.PointCloud(
        transform_npc(pc.to_array(), trf).astype(np.float32))
    new_pc.sensor_origin = (trf.orient.array.dot(pc.sensor_origin[:3]) +
                            trf.pos.array).astype(np.float32)
    new_pc.sensor_orientation = (
        (trf.orient * m3d.UnitQuaternion(*pc.sensor_orientation))
        .unit_quaternion.array).astype(np.float32)
    return new_pc


def crop_npc(npc, roi, roi_in_base=None):
    """Given a npy-point cloud and an 'roi' in the same coordinate
    reference, return the cropped point cloud. If 'roi_in_base' is
    given, it is assumed to represent the reference in which the ROI
    is given.
    """
    if roi_in_base is not None:
        # Transform the point cloud to ROI reference
        npc = transform_npc(npc, roi_in_base.inverse)
    npc_cropped = npc[(npc[:, 0] > roi[0][0]) & (npc[:, 0] < roi[0][1]) &
                      (npc[:, 1] > roi[1][0]) & (npc[:, 1] < roi[1][1]) &
                      (npc[:, 2] > roi[2][0]) & (npc[:, 2] < roi[2][1])]
    if roi_in_base is not None:
        # Transform back to base coordinates
        npc_cropped = transform_npc(npc_cropped, roi_in_base)
    return npc_cropped


def crop_pc(pc, roi, roi_in_base=None):
    """Given a pcl point cloud and an 'roi' in the same coordinate
    reference, return the cropped point cloud. If 'roi_in_base' is
    given, it is assumed to represent the reference in which the ROI
    is given.
    """
    cropped_pc = pcl.PointCloud(crop_npc(pc.to_array(), roi, roi_in_base))
    cropped_pc.sensor_origin = pc.sensor_origin
    cropped_pc.sensor_orientation = pc.sensor_orientation
    return cropped_pc


def projection_along(pc, direction, origin=None):
    """Calculate the projection of the cloud points along the given
    'direction'. If 'direction' is a unit vector, measurement units will
    be preserved."""
    if type(pc) == pcl.PointCloud:
        npc = pc.to_array()
    else:
        npc = pc
    if type(direction) == m3d.Vector:
        direction = direction.array
    if origin is not None:
        if type(origin) == m3d.Vector:
            origin = origin.array
        npc = npc - origin
    return npc.dot(direction)


def crop_along_npc(npc, origo, direction, limits):
    if type(origo) == m3d.Vector:
        origo = origo.array
    if type(direction) == m3d.Vector:
        direction = direction.array
    proj = (npc - origo).dot(direction)
    return npc[(proj > limits[0]) & (proj < limits[1])]


def crop_along_pc(pc, origo, direction, limits):
    """Crop the point cloud according to the 'limits', which is a pair,
    when projected on 'direction' and measured from position 'origo'.
    """
    if type(pc) == pcl.PointCloud:
        npc = pc.to_array()
    else:
        npc = pc
    cropped_npc = crop_along_npc(npc, origo, direction, limits)
    # Return the correct type
    if type(pc) == pcl.PointCloud:
        cropped_pc = pcl.PointCloud(cropped_npc)
        cropped_pc.sensor_origin = pc.sensor_origin
        cropped_pc.sensor_orientation = pc.sensor_orientation
    else:
        cropped_pc = cropped_npc
    return cropped_pc


crop_along = crop_along_pc


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
