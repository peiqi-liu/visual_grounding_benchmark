#!/usr/bin/env python

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(
    description="A simple script that takes an input file as an argument"
)

# Add the input file argument
parser.add_argument("--input_file", help="Path to the r3d input file")
parser.add_argument(
    "--output_file", default="pointcloud.ply", help="Path to the outputed ply file"
)
parser.add_argument(
    "--x1", default=None, type=float
)
parser.add_argument(
    "--x2", default=None, type=float
)
parser.add_argument(
    "--y1", default=None, type=float
)
parser.add_argument(
    "--y2", default=None, type=float
)
parser.add_argument(
    "--z_offset", default=None, type=float
)

# Parse the arguments
args = parser.parse_args()

assert (
    args.output_file[-4:] == ".ply"
), "The output file needs to be a ply file, which means the file name needs to end in .ply"

# Print the input file path
print(f"{args.input_file=}")
print(f"{args.output_file=}")

from a_star.data_util import get_pointcloud
from a_star.dataset_class import R3DDataset


dataset = R3DDataset(path = args.input_file, subsample_freq = 10, x1 = args.x1, x2 = args.x2, y1 = args.y1, y2 = args.y2, z_offset = args.z_offset)
get_pointcloud(
    dataset, args.output_file
    # get_posed_rgbd_dataset(key="r3d", path=args.input_file), args.output_file
)

print("... created " + args.output_file + ".")
