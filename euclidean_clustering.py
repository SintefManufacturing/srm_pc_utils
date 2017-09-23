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

import math3d as m3d
import math3d.geometry
import numpy as np
import scipy.spatial as spsp
import pcl


class EuclideanClusterExtractor:
    def __init__(self, nn_dist, nn_min=5, min_pts=5, max_pts=None,
                 min_max_length=None, max_max_length=None):
        self._log = logging.getLogger('ECE')
        self.nn_dist = nn_dist
        self.nn_min = nn_min
        self.min_pts = min_pts
        self.max_pts = max_pts
        self.min_max_length = min_max_length
        self.max_max_length = max_max_length

    def __call__(self, pc, filter=False):
        return self.extract(pc, filter)

    def extract(self, pc, filter=False, remainder=False):
        """Extract the clusters and return them as a list of separate point
        clouds. If filter flagged, do not return cluster but return
        instead the merge of all clusters.
        """
        npc = pc.to_array()
        clusters = []
        points = list(range(pc.size))
        proc_points = []
        queue = []
        self._log.debug('Building kdtree')
        kdt = spsp.cKDTree(npc)
        self._log.debug('... Done')
        while len(points) > 0:
            cpi = points.pop()
            c = [cpi]
            self._log.debug('New cluster {}'.format(len(clusters)))
            queue.append(cpi)
            while len(queue) > 0:
                pi = queue.pop()
                if pi in proc_points:                    
                    continue
                # self._log.debug('Analysing point {}'.format(pi))
                pinns = kdt.query_ball_point(npc[pi], self.nn_dist)
                if len(pinns) > self.nn_min:
                    c.append(pi)
                    proc_points.append(pi)
                    queue.extend(pinns)
            self._log.debug('Cluster size {}'.format(len(c)))
            if len(c) > self.min_pts:
                clusters.append(c)
            else:
                self._log.debug('Pre-rejecting small cluster.')
        acc_clusters = []
        for c in clusters:
            cpc = pc.extract(c)
            cnpc = cpc.to_array()
            if cpc.size < self.min_pts:
                self._log.debug('Rejecting cluster with {} points [min_pts]'.format(cpc.size))
                continue
            if self.max_pts is not None and cpc.size > self.max_pts:
                self._log.debug('Rejecting cluster with {} points [max_pts]'.format(cpc.size))
                continue
            if self.min_max_length is not None or self.max_max_length is not None:
                try:
                    hull = spsp.ConvexHull(cnpc)
                except:
                    self._log.warn('Hull computation failed')
                    continue
                max_length = spsp.distance.pdist(hull.points[hull.vertices]).max()
                if self.min_max_length is not None and max_length < self.min_max_length:
                    self._log.debug('Rejecting cluster of length {} [min_max_length]'.format(max_length))
                    continue
                if self.max_max_length is not None and max_length > self.max_max_length:
                    self._log.debug('Rejecting cluster of length {} [max_max_length]'.format(max_length))
                    continue
            acc_clusters.append(cpc)
        if filter:
            ncfpc = np.vstack([c.to_array() for c in acc_clusters])
            cfpc = pcl.PointCloud(ncfpc)
            cfpc.sensor_orientation = pc.sensor_orientation
            cfpc.sensor_origin = pc.sensor_origin
            return cfpc
        else:
            return acc_clusters


if __name__ == '__main__':
    import pcl
    logging.basicConfig(level=logging.DEBUG)
    ec = EuclideanClusterExtractor()
    pc = pcl.load('scene_cropped.pcd')
    ecs = ec.extract(pc, 0.005, min_pts=50, max_pts=500, min_max_length=0.03, max_max_length=0.09)
