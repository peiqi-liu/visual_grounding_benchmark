#!/usr/bin/env python

import argparse
import open3d as o3d
import pickle as pkl
import numpy as np
import torch

def numpy_to_pcd(xyz: np.ndarray, rgb: np.ndarray = None) -> o3d.geometry.PointCloud:
    """Create an open3d pointcloud from a single xyz/rgb pair"""
    xyz = xyz.reshape(-1, 3)
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

# Create the argument parser
parser = argparse.ArgumentParser(
    description="A simple script that takes an input file as an argument"
)

# Add the input file argument
parser.add_argument("--input_file", help="Path to the pkl input file")
parser.add_argument(
    "--steps", default=[], type=int, nargs='+'
)

# Parse the arguments
args = parser.parse_args()

# Print the input file path
print(f"{args.input_file=}")
print(f"{args.steps=}")

from home_robot.vision_pipeline import ImageProcessor

image_processor = ImageProcessor(open_communication = False, rerun = False, static = False)
with open(args.input_file, 'rb') as f:
    data = pkl.load(f)

for i, (
    camera_pose,
    rgb,
    depth,
    K,
) in enumerate(
    zip(
        data["camera_poses"],
        data["rgb"],
        data["depth"],
        data["camera_K"],
    )
):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().detach().numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().detach().numpy()
    if isinstance(K, torch.Tensor):
        K = K.cpu().detach().numpy()
    if isinstance(camera_pose, torch.Tensor):
        camera_pose = camera_pose.cpu().detach().numpy()
    image_processor.process_rgbd_images(rgb, depth, K, camera_pose)
    if i in args.steps:
        # points, _, _, rgb = image_processor.voxel_map.voxel_pcd.get_pointcloud()
        points, _, _, rgb = image_processor.voxel_map_localizer.voxel_pcd.get_pointcloud()
        points, rgb = points.detach().cpu().numpy(), rgb.detach().cpu().numpy()
        pcd = numpy_to_pcd(points, rgb / 255)
        o3d.io.write_point_cloud(args.input_file[:-4] + str(i) + '.pcd', pcd)
        print("... created " + args.input_file[:-4] + str(i) + '.pcd' + ".")
