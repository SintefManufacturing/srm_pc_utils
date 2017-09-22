# coding=utf-8

"""
"""

__author__ = "Morten Lind"
__copyright__ = "SINTEF 2017"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten.lind@sintef.no"
__status__ = "Development"

import logging
import time

import math3d as m3d
import math3d.geometry as geo
import numpy as np
import pcl
import scipy.spatial

class SACPlane:
    def __init__(self, eqn, hull, pc=None, obs_pose=None):
        self.eqn = eqn
        self.hull = hull
        self.hull_points = hull.points[hull.vertices]
        self.pc = pc
        self.obs_pose = obs_pose
        self.pl = geo.Plane(coeffs=eqn)
        self.centre = m3d.Vector(
            np.average(hull.points[hull.vertices], axis=0))
        self.normal = self.pl.normal
        if pc is not None and obs_pose is None:
            # If pc is given, and no observation pose, use the one
            # from the pc.
            self.obs_pose = m3d.Transform(
                m3d.UnitQuaternion(*pc.sensor_orientation).orientation,
                m3d.Vector(pc.sensor_origin[:3]))
        if self.obs_pose is not None:
            # Correct the normal to outward if we have an observation
            # pose
            if self.normal * (self.obs_pose.pos - self.centre) < 0.0:
                self.normal = -self.normal

    def dist(self, other):
        if type(other) is SACPlane:
            return self.dist(other.hull_points)
        elif type(other) == np.ndarray:
            if len(other.shape) == 2 and other.shape[1] == 3:
                # Assume an array of row vectors
                return min([self.dist(p) for p in other])
            elif len(other.shape) == 1 and other.shape[0] == 3:
                return min(
                    [np.linalg.norm(other - p) for p in self.hull_points])
            else:
                raise Exception('Can not measure dist to array of shape {}'
                                .format(other.shape))
        elif type(other) == m3d.Vector:
            return self.dist(other.array)
        else:
            raise Exception('Can not measure distance to object of type {}'
                            .format(type(other)))

    def __repr__(self):
        return 'Pln:@c{}n{}#{}'.format(
            self.centre.array, self.normal.array, len(self.hull.points))


class PlaneSegmenter:
    def __init__(self,
                 distance_tolerance=0.01,
                 axis=None,
                 perpend=True,
                 plane_normal_tolerance=0.3,
                 use_point_normals=True,
                 normal_distance_weight=0.001,
                 consume_distance=None,
                 maximum_iterations=10000,
                 minimum_plane_points=None,
                 minimum_plane_area=None
                 ):
        """Notes:

        The flag 'perpend' signals whether the *plane* to look for has
        it's normal perpendicular to 'axis'. If 'perpend' is False,
        planes for which 'axis' is parallel to the plane normal are
        searched for.

        If 'consume_distance' is given, this is the distance from the
        matched plane, within which points will b taken as belonging
        to the plane.

        Return value is a pair. The first element is a list of the
        matched SAC planes, the second element is the remainder point
        cloud.
        """
        self._dist_tol = distance_tolerance
        self._norm_dist_weight = normal_distance_weight
        self._normal_tol = plane_normal_tolerance
        self._axis = axis
        if self._axis is not None:
            self._axis = m3d.Vector(self._axis).normalized
        self._perpend = perpend
        self._use_normals = use_point_normals
        self._cons_dist = consume_distance
        self._max_iter = maximum_iterations
        self._min_pts = minimum_plane_points
        self._min_area = minimum_plane_area
        self._log = logging.getLogger('PlaneSegm')

    def __call__(self, pc):
        return self.segment(pc)

    def segment(self, pc):
        """Repeatedly extract planes from 'pc'. Return planes and remaining
        points.
        """
        t0 = time.time()
        # s_origin = pc.sensor_origin
        # s_orientation = pc.sensor_orientation
        # s_pose = m3d.Transform(m3d.UnitQuaternion(*s_orientation).orientation,
        #                        m3d.Vector(s_origin))
        planes = []
        plane_found = True
        while (pc.size > self._min_pts
               and plane_found):
            if self._use_normals:
                segm = pc.make_segmenter_normals(ksearch=10)
                segm.set_model_type(pcl.SACMODEL_PLANE)
                segm.set_normal_distance_weight(self._norm_dist_weight)
                segm.set_max_iterations(self._max_iter)
            else:
                segm = pc.make_segmenter()
                segm.set_model_type(pcl.SACMODEL_PLANE)
            segm.set_optimize_coefficients(True)
            segm.set_method_type(pcl.SAC_RANSAC)
            segm.set_distance_threshold(self._dist_tol)
            idx, model = segm.segment()
            # Check cardinality
            if len(idx) < self._min_pts:
                self._log.debug('No plane found (< {} pts)'.format(self._min_pts))
                plane_found = False
                break
            pc_pl = pc.extract(idx)
            pc_rem = pc.extract(idx, negative=True)
            hull = scipy.spatial.ConvexHull(pc_pl)
            sapl = SACPlane(model, hull, pc_pl)
            plchar = str(sapl)
            # # Check cardinality
            # if self._min_pts is not None and len(idx) < self._min_pts:
            #     self._log.debug('{} - min_points ({})'
            #                     .format(plchar, self._min_pts))
            #     pc = pc_rem
            #     continue
            # Check if correctly oriented
            if self._axis is not None:
                prod = np.abs(sapl.normal * self._axis)
                if self._perpend and prod > self._normal_tol:
                    self._log.debug('{} - perpendicular (>{})'
                                    .format(plchar, self._normal_tol))
                    pc = pc_rem
                    continue
                elif not self._perpend and prod < 1 - self._normal_tol:
                    self._log.debug('{} - parallel (<{})'
                                    .format(plchar, 1 - self._norm_tol))
                    pc = pc_rem
                    continue
            # Analyze hull size
            if self._min_area is not None and hull.area < self._min_area:
                self._log.debug('{} - area : {} m^2 (< {} m^2)'
                                .format(plchar, hull.area/2,
                                        self._min_face_area/10))
                pc = pc_rem
                continue
            self._log.debug('{} ACCEPT'.format(plchar))
            # All criteria fulfilled. If consume_distance is given,
            # then consume further points around the plane. TODO. See
            # sphere_based_he_calibration.sphere_recognition.exp
            if self._cons_dist is not None:
                # Move points in pc_rem to pc_pl, if they are within
                # cons_dist from the identified plane
                npvec = m3d.geometry.Plane(coeffs=model).plane_vector
                pdists = np.abs(npvec.array.dot(pc.to_array().T) - 1)
                xpidx = np.where(pdists < self._cons_dist)[0]
                pc_pl = pc.extract(xpidx)
                pc_rem = pc.extract(xpidx, negative=True)
                hull = scipy.spatial.ConvexHull(pc_pl)
                sapl = SACPlane(model, hull, pc_pl)
                self._log.debug('SACPlane extended to #{}'.format(pc_pl.size))
            planes.append(sapl)
            pc = pc_rem
        self._log.debug('{} acceptable planes found in {:.3f} s'
                        .format(len(planes), time.time()-t0))
        return (planes, pc)
