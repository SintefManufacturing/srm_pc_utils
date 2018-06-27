#!/usr/bin/env python3
# coding=utf-8

"""
Commandline for voxel grid filtering
"""

__author__ = "Morten Lind"
__copyright__ = "SINTEF 2018"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten.lind@sintef.no"
__status__ = "Development"

import argparse

import pcl


ap = argparse.ArgumentParser(
    description=
    """Commandline utility for voxel grid filtering (downsampling) of a
    point cloud.""")
ap.add_argument('input_pc', type=str, help='Input point cloud.')
ap.add_argument('voxel_size', type=float, help='Voxel size in scene units.')
ap.add_argument('output_pc', type=str, help='Output point cloud.')
args = ap.parse_args()

print(args)
pc = pcl.load(args.input_pc)
vgf = pc.make_voxel_grid_filter()
vgf.set_leaf_size(*3*[args.voxel_size])
pcl.save(vgf.filter(), args.output_pc)
