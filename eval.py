from stretch.perception.detection.owl import OwlPerception
from stretch.perception.encoders import MaskSiglipEncoder
from stretch.perception.encoders.masksiglip2_encoder import MaskSiglip2Encoder
from stretch.perception.detection.yoloe import YoloEPerception
from stretch.perception.encoders.mobile_clip_encoder import MaskMobileClipEncoder
from stretch.perception.encoders.clip_encoder import MaskClipEncoder
from stretch.mapping.voxel import SparseVoxelMapDynamem as SparseVoxelMap

import pandas as pd
import numpy as np
import torch
import pickle as pkl
import argparse

import rerun as rr

import time

all_env_non_exist_total = 0
all_env_non_exist_correct = 0
all_env_exist_correct = 0
all_env_exist_total = 0
all_env_exist_mistakenly_localized = 0 
all_env_exist_not_localized = 0

def process_csv(csv_filename: str):
    '''
        Util function for processing csv file
    '''
    annotations = pd.read_csv(csv_filename)
    queries, xs, ys, zs, tols = annotations['query'].values, annotations['x'].values, annotations['y'].values, annotations['z'].values, annotations['tol'].values
    exist_index = ~np.isnan(xs)
    exist_queries = queries[exist_index]
    non_exist_queries = queries[~exist_index]
    xs, ys, zs, tols = xs[exist_index], ys[exist_index], zs[exist_index], tols[exist_index]
    xyzs = np.stack([xs, ys, zs]).T
    return exist_queries, xyzs, tols, non_exist_queries
    

def read_time_steps(folder_name: str) -> list[int]:
    time_steps = []
    with open(folder_name + '/time.txt', 'r') as f:
        for line in f:
            time_steps.append(int(line.strip()))
    return time_steps

def eval_one_time_step(folder_name: str, time_step: int, voxel_map):
    global all_env_non_exist_total, all_env_non_exist_correct, all_env_exist_correct, all_env_exist_total, all_env_exist_mistakenly_localized, all_env_exist_not_localized

    rr.init('Eval ' + folder_name + ' ' + str(time_step), spawn = True)

    score = 0
    # print(folder_name, time_step)
    csv_filename = folder_name + '/' + str(time_step) + '.csv'
    exist_queries, xyzs, tols, non_exist_queries = process_csv(csv_filename)
    not_localized, mistakenly_localized, exist_total, exist_correct = 0, 0, 0, 0

    # if voxel_map.semantic_memory._points is not None:
    #         rr.log("Semantic_memory/pointcloud", rr.Points3D(voxel_map.semantic_memory._points.detach().cpu(), \
    #                     colors=voxel_map.semantic_memory._rgb.detach().cpu() / 255., radii=0.03))
    if voxel_map.voxel_pcd._points is not None:
            rr.log("Semantic_memory/pointcloud", rr.Points3D(voxel_map.voxel_pcd._points.detach().cpu(), \
                        colors=voxel_map.voxel_pcd._rgb.detach().cpu() / 255., radii=0.03))

    import os 
    import cv2

    for query, xyz, tol in zip(exist_queries, xyzs, tols):
        pred_xyz, debug_text = voxel_map.localize_text(query, debug = True, return_debug = False)

        if pred_xyz is not None:
            rr.log(query.replace(' ', '_') + '/predicted', rr.Points3D([pred_xyz[0], pred_xyz[1], pred_xyz[2]], colors=torch.Tensor([1, 0, 0]), radii=0.1))
        rr.log(query.replace(' ', '_') + '/labeled', rr.Points3D([xyz[0], xyz[1], xyz[2]], colors=torch.Tensor([0, 1, 0]), radii=0.1))
        rr.log(query.replace(' ', '_'), rr.TextDocument(debug_text, media_type = rr.MediaType.MARKDOWN))
        obs_id = int(voxel_map.find_obs_id_for_text(query).detach().cpu().item())
        rgb = voxel_map.observations[obs_id - 1].rgb
        rr.log(query.replace(' ', '_') + '/Memory_image', rr.Image(rgb))

        exist_total += 1
        if pred_xyz is not None and np.linalg.norm(pred_xyz - xyz) <= tol:
            exist_correct += 1
            score += 1
            print(query, 'is correctly found.')
        elif pred_xyz is not None:
            mistakenly_localized += 1
            print(query, 'is mistakenly localized somewhere else.')
        else:
            not_localized += 1
            print(query, 'is not found.')
    print('\nExisting objects check', exist_correct, '/', exist_total)
    print(mistakenly_localized, 'objects are mistakenly localized somewhere else.')
    print(not_localized, 'objects are not found anywhere.', '\n')

    non_exist_total, non_exist_correct = 0, 0
    for query in non_exist_queries:
        pred_xyz, debug_text = voxel_map.localize_text(query, debug = True, return_debug = False)
            
        if pred_xyz is not None:
            rr.log(query.replace(' ', '_') + '/predicted', rr.Points3D([pred_xyz[0], pred_xyz[1], pred_xyz[2]], colors=torch.Tensor([1, 0, 0]), radii=0.1))
        rr.log(query.replace(' ', '_'), rr.TextDocument(debug_text, media_type = rr.MediaType.MARKDOWN))
        obs_id = int(voxel_map.find_obs_id_for_text(query).detach().cpu().item())
        rgb = voxel_map.observations[obs_id - 1].rgb
        rr.log(query.replace(' ', '_') + '/Memory_image', rr.Image(rgb))

        non_exist_total += 1
        if pred_xyz is None:
            score += 1
            non_exist_correct += 1
            print(query, 'is not found, just as expected')
        else:
            print(query, 'is found, which is incorrect as the object should have not been observed.')
    print('\nNon-existing objects check', non_exist_correct, '/', non_exist_total)
    print(non_exist_total - non_exist_correct, 'objects are found, which is incorrect as the object should have not been observed.', '\n')

    all_env_non_exist_total += non_exist_total
    all_env_non_exist_correct += non_exist_correct
    all_env_exist_correct += exist_correct
    all_env_exist_mistakenly_localized += mistakenly_localized
    all_env_exist_not_localized += not_localized
    all_env_exist_total += exist_total

    return exist_correct + non_exist_correct, exist_total + non_exist_total, score
    # return None, None, None

def eval_one_environment(folder_name: str, method: str):
    first = True
    total_score = 0
    
    # encoder = MaskSiglipEncoder(device="cuda", version="so400m")
    # encoder = MaskSiglip2Encoder(device="cuda", version="so400m")
    # detection_model = YoloEPerception(confidence_threshold=0.05, size="l")
    # encoder = MaskClipEncoder(device="cuda", version="ViT-B/16")
    encoder = MaskMobileClipEncoder(device="cuda", version="S2")
    detection_model = OwlPerception(version="owlv2-L-p14-ensemble", device="cuda", confidence_threshold=0.1)
    voxel_map = SparseVoxelMap(encoder = encoder, detection = detection_model, mllm=False, image_shape = (480, 360))

    time_steps = read_time_steps(folder_name)
    pkl_name = folder_name + '/env.pkl'
    with open(pkl_name, 'rb') as f:
        data = pkl.load(f)
    total_correct, total_queries = 0, 0
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
        import time
        start_time = time.time()
        voxel_map.process_rgbd_images(rgb, depth, K, camera_pose)
        end_time = time.time()
        # print('Image processing takes', end_time - start_time, 'seconds.')
        if i in time_steps:
            correct, total, score = eval_one_time_step(folder_name, i, voxel_map)
            print('Environment:', folder_name, 'time step:', i, 'testing result:', correct, '/', total, '=', correct * 1.0 / total)
            total_correct += correct
            total_queries += total
            if first:
                first = False
                total_score += 0.5 * score
            else:
                total_score += score
        # time.sleep(5)
    return total_correct, total_queries, total_score


parser = argparse.ArgumentParser(
    description="A simple script that takes an input file as an argument"
)

# Add the input file argument
parser.add_argument(
    "--folders", default=[], type=str, nargs='+'
)
parser.add_argument(
    "--method", type=str
)
parser.add_argument(
    "--seed", type=int, default=1
)

# Parse the arguments
args = parser.parse_args()
total_score = 0
assert len(args.folders) != 0, "No folder specified"

envs_success_rate = []
total_correct, total_queries = 0, 0
for folder_name in args.folders:
    torch.manual_seed(args.seed)
    correct, total, score = eval_one_environment(folder_name, args.method)
    print('Environment:', folder_name, 'testing result:', correct, '/', total, '=', correct * 1.0 / total)
    print('Environment:', folder_name, 'score:', score)
    envs_success_rate.append(correct * 1.0 / total)
    total_correct += correct
    total_queries += total
    total_score += score
print('Total rate:', total_correct, '/', total_queries, '=', total_correct * 1.0 / total_queries)
print('Total score:', total_score)

print(envs_success_rate)
print(all_env_non_exist_total, all_env_non_exist_correct, all_env_exist_total, all_env_exist_correct, all_env_exist_mistakenly_localized, all_env_exist_not_localized)
        
