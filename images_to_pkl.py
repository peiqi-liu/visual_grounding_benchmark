import argparse
import pickle as pkl
import numpy as np
import cv2

# Create the argument parser
parser = argparse.ArgumentParser(
    description="A simple script that takes an input file as an argument"
)

# Add the input file argument
parser.add_argument("--folder_name")
parser.add_argument("--image_num", type=int)

# Parse the arguments
args = parser.parse_args()

data = dict()
for i in range(1, args.image_num + 1):
    rgb = np.load(args.folder_name + '/rgb' + str(i) + '.npy')
    depth = np.load(args.folder_name + '/depth' + str(i) + '.npy')
    intrinsics = np.load(args.folder_name + '/intrinsics' + str(i) + '.npy')
    pose = np.load(args.folder_name + '/pose' + str(i) + '.npy')
    cv2.imwrite(args.folder_name + '/' + str(i) + '.jpg', rgb[:, :, [2, 1, 0]])
    data['rgb'] = data['rgb'] + [rgb] if 'rgb' in data else [rgb]
    data['depth'] = data['depth'] + [depth] if 'depth' in data else [depth]
    data['camera_K'] = data['camera_K'] + [intrinsics] if 'camera_K' in data else [intrinsics]
    data['camera_poses'] = data['camera_poses'] + [pose] if 'camera_poses' in data else [pose]

with open('output.pkl', 'wb') as file:
    pkl.dump(data, file)