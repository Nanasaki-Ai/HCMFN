import math
import os
import cv2
import sys
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
two_person_action = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
anchor = [3, 7, 11, 15, 19]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def read_xy(file, max_body=2, num_joint=25):  
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 2))   
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    if np.isnan(v['depthX']):
                        v['depthX'] = 0.0
                    if np.isnan(v['depthY']):
                        v['depthY'] = 0.0
                    data[m, n, j, :] = [v['depthX'], v['depthY']]
                else:
                    pass
    return data


def crop_box(x, y, delta_x, delta_y, person = 0):
    if person == 1:
        box = [x - delta_x/2, y - delta_y, x + delta_x/2, y + delta_y]
    elif person == 2:
        box = [x - delta_x/2, y - delta_y, x + delta_x/2, y + delta_y]
    else:
        box = [x - delta_x, y - delta_y, x + delta_x, y + delta_y]
    for i in range(len(box)):
        box[i] = math.floor(box[i])
    return box


def img_crop(imgs, box):
    img = imgs[box[1]:box[3], box[0]:box[2]]
    if img.shape == (96,48) or img.shape == (96,96):
        return img
    a = 550
    img = np.pad(imgs, (a, a), 'constant')
    img = img[box[1]+a:box[3]+a, box[0]+a:box[2]+a]
    if img.shape != (96, 48) and img.shape != (96, 96):
        img = np.zeros([box[3] - box[1], box[2] - box[0]])

    return img


def extract_frames_and_crop(frames_path, file_skeleton, images_out_path, delta, num_roi=5, num_f=5):
    skeleton_data = read_xy(file_skeleton, 2, 25)
    skeleton_2D = skeleton_data
    final_frame_index = skeleton_data.shape[1]
    sampling_interval = final_frame_index // num_f
    start_frame_index = final_frame_index - sampling_interval * num_f
    volume =0
    new = np.zeros([delta*2*num_f, delta*2*num_roi])
    for n_f in range(start_frame_index + sampling_interval -1, final_frame_index, sampling_interval):
        img = cv2.imread(frames_path + "/MDepth-%08d" % (n_f) + ".png", -1)
        
        if np.sum(skeleton_2D[1, :]) == 0:
            for n_roi in range(0, num_roi):
                box = crop_box(skeleton_data[0, n_f, anchor[n_roi], 0], skeleton_data[0, n_f, anchor[n_roi], 1], delta, delta)
                roi = img_crop(img, box)
                new[n_roi * delta * 2:n_roi * delta * 2 + delta * 2, volume * delta * 2:volume * delta * 2 + delta * 2] = roi
            volume = volume + 1
        else:
            for n_roi in range(0, num_roi):
                box1 = crop_box(skeleton_data[0, n_f, anchor[n_roi], 0], skeleton_data[0, n_f, anchor[n_roi], 1], delta, delta, person=1)
                box2 = crop_box(skeleton_data[1, n_f, anchor[n_roi], 0], skeleton_data[1, n_f, anchor[n_roi], 1], delta, delta, person=2)
                roi1 = img_crop(img, box1)
                roi2 = img_crop(img, box2)

                new[n_roi * delta * 2 : n_roi * delta * 2 + delta * 2,   volume * delta * 2 + delta: volume * delta * 2 + delta * 2] = roi2
                new[n_roi * delta * 2 : n_roi * delta * 2 + delta * 2,   volume * delta * 2 : volume * delta * 2 + delta] = roi1
            volume = volume + 1

    out_folder = os.path.join(images_out_path, frames_path.split('/')[-1])
    np.save(out_folder + '.npy', new)


def gendata(arg, label_out_path, images_out_path, benchmark='xview', part='eval'):
    if arg.ignored_sample_path != None:
        with open(arg.ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() for line in f.readlines()]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(arg.frames_path):
        if filename in ignored_samples:
            continue
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(label_out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    for s in tqdm(sample_name):
        skeleton_data_path = arg.skeletons_path + s + '.skeleton'
        extract_frames_and_crop(arg.frames_path + s, skeleton_data_path, images_out_path, 48, 5, 5)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--skeletons_path', default='E:/PyTorchTest/2sAGCN/data/nturgbd_raw/nturgb+d_skeletons/')  
    parser.add_argument('--frames_path', default='F:/NTU RGB+D/masked_depth_maps/nturgb+d_depth_masked/')  
    parser.add_argument('--ignored_sample_path', default='E:/PyTorchTest/2sAGCN/data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='ntu60_depth_hci')  

    benchmark = ['xsub']
    part = ['train', 'val']
    arg = parser.parse_args()
    for b in benchmark:
        for p in part:
            label_out_path = os.path.join(arg.out_folder, b)
            images_out_path = os.path.join(label_out_path, p)
            if not os.path.exists(images_out_path):
                os.makedirs(images_out_path)
            print(b, p)
            gendata(arg, label_out_path, images_out_path, benchmark=b, part=p)