from a_star.dataset_class import R3DDataset
import torch
import pickle as pkl
import argparse

def get_xyz_coordinates(depth, pose, intrinsic):

    _, height, width = depth.shape

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(width, device=depth.device),
        torch.arange(height, device=depth.device),
        indexing="xy",
    )

    x = (xs - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (ys - intrinsic[1, 2]) / intrinsic[1, 1]

    # Depth array should be the same shape as x and y
    z = depth[0]

    # Prepare camera coordinates
    camera_coords = torch.stack((x * z, y * z, z, torch.ones_like(z)), axis=-1)

    # Prepare pose matrix for broadcasting
    # Transform to world coordinates using the pose matrix
    world_coords = camera_coords @ pose.T

    # Return world coordinates (excluding the homogeneous coordinate)
    return world_coords[..., :3], camera_coords[..., :3]

# Create the argument parser
parser = argparse.ArgumentParser(
    description="A simple script that takes an input file as an argument"
)

parser.add_argument("--input_file", type=str)
# Parse the arguments
args = parser.parse_args()
data = dict()

dataset = R3DDataset(path = args.input_file, subsample_freq = 10)
    
for i in dataset:
    image, depth, _, intrinsics, camera_pose = i
    image = (image * 255).to(torch.uint8).permute(1, 2, 0)
    data['rgb'] = data['rgb'] + [image] if 'rgb' in data else [image]
    data['depth'] = data['depth'] + [depth[0]] if 'depth' in data else [depth[0]]
    data['camera_K'] = data['camera_K'] + [intrinsics] if 'camera_K' in data else [intrinsics]
    data['camera_poses'] = data['camera_poses'] + [camera_pose] if 'camera_poses' in data else [camera_pose]
with open(args.input_file[:-4] + '.pkl', 'wb') as file:
    pkl.dump(data, file)
