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
        self.density = 2 * pc.size / hull.area 
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
        return 'Pln:@c{}n{} #{} d{:.0f}'.format(
            self.centre.array, self.normal.array, self.pc.size, self.density)


class PlaneSegmenter:
    def __init__(self,
                 distance_tolerance=0.001,
                 use_point_normals=True,
                 normal_distance_weight=0.01,
                 axis=None,
                 perpend=False,
                 maximum_iterations=10000,
                 plane_normal_tolerance=0.3,
                 minimum_plane_points=10,
                 minimum_plane_area=0,
                 minimum_density=None,
                 consume_distance=None,
                 noise=0.0
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

        If 'noise' is given, this amount of uniform noise is added to
        the point cloud prior to processing. This may remove problems
        with instability of convex hull calculation and
        stratification.
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
        self._min_face_pts = minimum_plane_points
        self._min_face_area = minimum_plane_area
        self._min_density = minimum_density
        self._noise = noise
        self._log = logging.getLogger('PlaneSegm')

    def __call__(self, pc):
        return self.segment(pc)

    def segment(self, pc):
        """Repeatedly extract planes from 'pc'. The return value is a pair: a
        list of matched planes contained in SACPlane objects, and the
        un-consumed point cloud, that is the remainder of the point
        cloud which was not consumed by the matching planes.
        """
        self._log.debug('PC SORIGIN: %s' % str(pc.sensor_origin))
        t0 = time.time()
        # s_origin = pc.sensor_origin
        # s_orientation = pc.sensor_orientation
        # s_pose = m3d.Transform(m3d.UnitQuaternion(*s_orientation).orientation,
        #                        m3d.Vector(s_origin))
        # The matching planes
        planes = []
        # The unprocessed points in the point cloud decreases
        # monotonically as planes are found, whether they match or
        # not. Optionally add noise to avoid instablility.
        if self._noise is not None and self._noise > 0.0:
            pc_unprocessed = pcl.PointCloud(
                (pc.to_array() +
                 self._noise * (2*np.random.random((pc.size, 3))-1))
                .astype(np.float32))
            pc_unprocessed.sensor_origin = pc.sensor_origin
            pc_unprocessed.sensor_orientation = pc.sensor_orientation
        else:
            pc_unprocessed = pc

        # The points that have not been consumed by the matching
        # planes steadily increase, as non-matching planes are
        # found. At the end, the remaining unprocessed points are
        # added.
        npc_unconsumed = np.array([], dtype=np.float32).reshape(0,3)
        # Stop flag is whether a plane search was unsuccesful.
        plane_found = True
        while (pc_unprocessed.size > self._min_face_pts
               and plane_found):
            # Make a segmenter
            if self._use_normals:
                segm = pc_unprocessed.make_segmenter_normals(ksearch=10)
                segm.set_model_type(pcl.SACMODEL_PLANE)
                segm.set_normal_distance_weight(self._norm_dist_weight)
                segm.set_max_iterations(self._max_iter)
            else:
                segm = pc_unprocessed.make_segmenter()
                segm.set_model_type(pcl.SACMODEL_PLANE)
            segm.set_optimize_coefficients(True)
            segm.set_method_type(pcl.SAC_RANSAC)
            segm.set_distance_threshold(self._dist_tol)
            # Apply segmenter
            idx, model = segm.segment()
            self._log.debug('Extracting {} points'.format(len(idx)))
            # Check cardinality
            if len(idx) < self._min_face_pts:
                self._log.debug('No plane found (< {} pts)'.format(self._min_face_pts))
                plane_found = False
                break
            # Extract points
            pc_cand = pc_unprocessed.extract(idx)
            # Form the convex hull to analyze area and point
            # density. N.B.: We are matching a 3D convex hull on a
            # point set which is essentially planar. This is not
            # stable, so some noise is added to make sure there is
            # some volume.
            hull = scipy.spatial.ConvexHull(pc_cand.to_array())
            sapl = SACPlane(model, hull, pc_cand)
            plchar = str(sapl)
            # Flag when a test rejects the candidate plane
            rejected = False
            # Check density
            if not rejected and self._min_density is not None:
                if sapl.density < self._min_density:
                    self._log.debug('{} - density ({}<{})'
                                    .format(plchar, sapl.density, self._min_density))
                    rejected = True
            # Check oriented
            if not rejected and self._axis is not None:
                prod = np.abs(sapl.normal * self._axis)
                if self._perpend and prod > self._normal_tol:
                    self._log.debug('{} - perpendicular (>{})'
                                    .format(plchar, self._normal_tol))
                    rejected = True
                elif not self._perpend and prod < 1 - self._normal_tol:
                    self._log.debug('{} - parallel (<{})'
                                    .format(plchar, 1 - self._normal_tol))
                    rejected = True
            # Analyze hull size. Note that the area of a "flat" hull
            # is twice the area of one side.
            if not rejected and self._min_face_area is not None and hull.area / 2 < self._min_face_area:
                self._log.debug('{} - area : {} m^2 (< {} m^2)'
                                .format(plchar, hull.area/2,
                                        self._min_face_area))
                rejected = True
            # Process rejected planes
            if rejected:
                # Rejected planes are un-consumed and processed
                self._log.debug('Marking as un-consumed')
                npc_unconsumed = np.vstack((npc_unconsumed, pc_cand.to_array()))
                pc_unprocessed = pc_unprocessed.extract(idx, negative=True)
                continue
            else:
                # The plane is to be consumed
                self._log.debug('{} ACCEPT'.format(plchar))
            # All criteria fulfilled. If consume_distance is given,
            # then consume further points around the plane.
            if self._cons_dist is not None:
                # Re-compute a pc_cand from the plane model and select
                # extended point set based on distance, cutting off at
                # cons_dist
                #npvec = m3d.geometry.Plane(coeffs=model).plane_vector
                #self._log.debug('Normalized plane vector length: {}'.format(npvec.length))
                #pdists = np.abs(npvec.array.dot(pc_unprocessed.to_array().T) - 1)
                plp, pln = m3d.geometry.Plane(coeffs=model).point_normal
                pdists = np.abs(pln.array.dot((pc_unprocessed.to_array() - plp.array).T))
                xpidx = np.where(pdists < self._cons_dist)[0]
                if len(xpidx) > 0:
                    self._log.debug('Consuming {} extra points'.format(len(xpidx) - len(idx)))
                    pc_xcand = pc_unprocessed.extract(xpidx)
                    pc_unprocessed = pc_unprocessed.extract(xpidx, negative=True)
                    pc_cand = pcl.PointCloud(np.vstack((pc_cand.to_array(), pc_xcand.to_array())))
                    pc_cand.sensor_origin = pc.sensor_origin
                    pc_cand.sensor_orientation = pc.sensor_orientation
                    hull = scipy.spatial.ConvexHull(pc_cand.to_array())
                    sapl = SACPlane(model, hull, pc_cand)
                    self._log.debug('SACPlane extended to #{}'.format(pc_cand.size))
                else:
                    self._log.debug('No extra points to consume')
            else:
                # Only extract the plane points from the unprocessed
                pc_unprocessed = pc_unprocessed.extract(idx, negative=True)
            planes.append(sapl)
        # Remaining unprocessed points are un-consumed
        npc_unconsumed = np.vstack((npc_unconsumed, pc_unprocessed.to_array()))
        pc_unconsumed = pcl.PointCloud(npc_unconsumed)
        pc_unconsumed.sensor_origin = pc.sensor_origin
        pc_unconsumed.sensor_orientation = pc.sensor_orientation
        self._log.debug('{} acceptable planes found in {:.3f} s'
                        .format(len(planes), time.time()-t0))
        return planes, pc_unconsumed
