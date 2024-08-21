from a_star.dataset_class import R3DDataset
import torch
import pickle as pkl
import argparse
import random

# import cv2

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

data = dict()
data['base_poses'] = []
data['feats'] = []
data['obs'] = []
data['xyz'] = []
data['world_xyz'] = []

# Create the argument parser
parser = argparse.ArgumentParser(
    description="A simple script that takes an input file as an argument"
)

parser.add_argument("--input_file", default=[], type=str, nargs='+')
parser.add_argument(
    "--x1", default=[], type=float, nargs='+'
)
parser.add_argument(
    "--x2", default=[], type=float, nargs='+'
)
parser.add_argument(
    "--y1", default=[], type=float, nargs='+'
)
parser.add_argument(
    "--y2", default=[], type=float, nargs='+'
)
parser.add_argument(
    "--z_offset", default=[], type=float, nargs='+'
)
# Parse the arguments
args = parser.parse_args()

# count = 0

print(args.input_file, args.x1, args.x2, args.y1, args.y2, args.z_offset)

for (r3d, x1, x2, y1, y2, z_offset) in zip(args.input_file, args.x1, args.x2, args.y1, args.y2, args.z_offset):
    print(r3d, x1, x2, y1, y2, z_offset)
    dataset = R3DDataset(path = r3d, subsample_freq = 20, x1 = x1, x2 = x2, y1 = y1, y2 = y2, z_offset = z_offset)
    print(len(dataset))
    r3d_rgb, r3d_depth, r3d_K, r3d_pose = [], [], [], []
    
    for i in dataset:
        image, depth, _, intrinsics, camera_pose = i
        image = (image * 255).to(torch.uint8).permute(1, 2, 0)
        r3d_rgb.append(image)
        # cv2.imwrite('debug_hardware_lab/' + str(count) + '.jpg', image.numpy()[:, :, [2, 1, 0]])
        # count += 1
        r3d_depth.append(depth[0])
        r3d_K.append(intrinsics)
        r3d_pose.append(camera_pose)
    # zipped_list = list(zip(r3d_depth, r3d_rgb, r3d_K, r3d_pose))
    # random.shuffle(zipped_list)
    # r3d_depth, r3d_rgb, r3d_K, r3d_pose = zip(*zipped_list)
    # print(len(r3d_depth))
    
    data['rgb'] = data['rgb'] + r3d_rgb if 'rgb' in data else r3d_rgb
    data['depth'] = data['depth'] + r3d_depth if 'depth' in data else r3d_depth
    data['camera_K'] = data['camera_K'] + r3d_K if 'camera_K' in data else r3d_K
    data['camera_poses'] = data['camera_poses'] + r3d_pose if 'camera_poses' in data else r3d_pose
    del dataset
with open('env.pkl', 'wb') as file:
    pkl.dump(data, file)
    del data
