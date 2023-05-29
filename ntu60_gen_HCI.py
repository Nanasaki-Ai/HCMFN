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
import shutil
import torch
from flownet2.models import FlowNet2SD
from torchvision.transforms import Resize

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

torch_resize = Resize([64,64])
torch_resize2 = Resize([64,32])

def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
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
                    if np.isnan(v['colorX']):
                        v['colorX'] = 0.0
                    if np.isnan(v['colorY']):
                        v['colorY'] = 0.0
                    data[m, n, j, :] = [v['colorX'], v['colorY']]
                else:
                    pass
    return data


def Generate_Optical_Flow(img, f_img, Two=False):
    this_img = np.array(img)  # 1080,1920,3  0-255
    forward_img = np.array(f_img)
    this_img = this_img.transpose(2, 0, 1)  # 1080,1920,3  0-255
    forward_img = forward_img.transpose(2, 0, 1)
    this_img = np.expand_dims(this_img, axis=0)
    forward_img = np.expand_dims(forward_img, axis=0)
    this_img = torch.tensor(this_img)
    forward_img = torch.tensor(forward_img)
    this_img = torch_resize(this_img)
    forward_img = torch_resize(forward_img)
    flow_input = torch.cat([forward_img.unsqueeze(2), this_img.unsqueeze(2)], 2).cuda() 
    if Two:
        flow = (flow_net(flow_input * 1.0))
        flow = torch_resize2(flow).cpu().detach().numpy()
    else:
        flow = (flow_net(flow_input * 1.0)).cpu().detach().numpy() 
    flow = flow.transpose(0, 2, 3, 1)
    return flow


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


def extract_frames_and_crop(frames_path, file_skeleton, images_out_path, opt_out_path, delta, num_roi=5, num_f=5):
    skeleton_data = read_xy(file_skeleton, 2, 25)
    skeleton_2D = skeleton_data

    final_frame_index = skeleton_data.shape[1]
    sampling_interval = final_frame_index // num_f
    start_frame_index = final_frame_index - sampling_interval * num_f
    volume =0
    
    new = np.zeros([delta*2*num_f, delta*2*num_roi, 3])    
    new_opt = np.zeros([32 * 2 * num_f, 32 * 2 * num_roi, 2])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    temporal_dir = os.path.join(current_dir, 'temporal')
    if not os.path.exists(temporal_dir):
        os.makedirs(temporal_dir)    
        
    video_number = file_skeleton.split('/')[-1].split('.')[0]
    video_path = frames_path + "/" + video_number + "_rgb" + ".avi"

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("can not read video files")
        
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break       
        file_name = os.path.join(temporal_dir, f"{frame_count}.jpg")
        cv2.imwrite(file_name, frame)
        frame_count += 1
    video.release()
    
    
    for n_f in range(start_frame_index + sampling_interval -1, final_frame_index, sampling_interval):     
        img_path = os.path.join(temporal_dir, f"{n_f}.jpg")
        img = Image.open(img_path)
        
        f_img_path = os.path.join(temporal_dir, f"{n_f - sampling_interval + 1}.jpg")
        f_img = Image.open(f_img_path)
               
        if np.sum(skeleton_2D[1, :]) == 0:
            for n_roi in range(0, num_roi):
                box = crop_box(skeleton_data[0, n_f, anchor[n_roi], 0], skeleton_data[0, n_f, anchor[n_roi], 1], delta, delta)
                roi = img.crop(box)
                f_roi = f_img.crop(box)
                f_roi = Generate_Optical_Flow(roi, f_roi)
                new_opt[n_roi * 32 * 2:n_roi * 32 * 2 + 32 * 2, volume * 32 * 2:volume * 32 * 2 + 32 * 2, :] = f_roi                
                new[n_roi * delta * 2:n_roi * delta * 2 + delta * 2, volume * delta * 2:volume * delta * 2 + delta * 2, :] = roi
            volume = volume + 1
                                
        else:
            for n_roi in range(0, num_roi):
                box1 = crop_box(skeleton_data[0, n_f, anchor[n_roi], 0], skeleton_data[0, n_f, anchor[n_roi], 1], delta, delta, person=1)
                box2 = crop_box(skeleton_data[1, n_f, anchor[n_roi], 0], skeleton_data[1, n_f, anchor[n_roi], 1], delta, delta, person=2)
                roi1 = img.crop(box1)
                roi2 = img.crop(box2)
                f_roi1 = f_img.crop(box1)
                f_roi2 = f_img.crop(box2)
                
                new[n_roi * delta * 2 : n_roi * delta * 2 + delta * 2,   volume * delta * 2 + delta: volume * delta * 2 + delta * 2, :] = roi2
                new[n_roi * delta * 2 : n_roi * delta * 2 + delta * 2,   volume * delta * 2 : volume * delta * 2 + delta, :] = roi1
                f_roi1 = Generate_Optical_Flow(roi1, f_roi1, True)
                f_roi2 = Generate_Optical_Flow(roi2, f_roi2, True)
                new_opt[n_roi * 32 * 2 : n_roi * 32 * 2 + 32 * 2,   volume * 32 * 2 + 32: volume * 32 * 2 + 32 * 2, :] = f_roi2
                new_opt[n_roi * 32 * 2 : n_roi * 32 * 2 + 32 * 2,   volume * 32 * 2 : volume * 32 * 2 + 32, :] = f_roi1
                
            volume = volume + 1

    out_folder = os.path.join(images_out_path, video_number)
    new = cv2.cvtColor(new.astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_folder + '.jpg', new)
    
    out_folder_opt = os.path.join(opt_out_path, video_number)
    np.save(out_folder_opt + '.npy', new_opt)
    
    shutil.rmtree(temporal_dir)

def gendata(arg, label_out_path, images_out_path, opt_out_path, benchmark='xsub', part='train'):
    if arg.ignored_sample_path != None:
        with open(arg.ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() for line in f.readlines()]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(arg.skeletons_path):

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

    # with open('{}/{}_label.pkl'.format(label_out_path, part), 'wb') as f:
    #     pickle.dump((sample_name, list(sample_label)), f)

    for s in tqdm(sample_name):
        skeleton_data_path = arg.skeletons_path + s
        extract_frames_and_crop(arg.frames_path, skeleton_data_path, images_out_path, opt_out_path, 48, 5, 5)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--skeletons_path', default='E:/PyTorchTest/2sAGCN/data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--frames_path', default='rgb video/nturgb+d_rgb/')
    parser.add_argument('--ignored_sample_path', default='E:/PyTorchTest/2sAGCN/data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='ntu60_hci')
    parser.add_argument('--out_folder_opt', default='ntu60_opt_hci')


    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('flownet2/FlowNet2-SD.pth')['state_dict'])
    flow_net.eval().cuda()
    
    benchmark = ['xsub']
    part = ['train', 'val']
    arg = parser.parse_args()
    for b in benchmark:                
        for p in part:
            label_out_path = os.path.join(arg.out_folder, b)
            images_out_path = os.path.join(arg.out_folder, 'train')
            if not os.path.exists(images_out_path):
                os.makedirs(images_out_path)
            # opt_label_out_path = os.path.join(arg.out_folder_opt, b)
            opt_out_path = os.path.join(arg.out_folder_opt, 'train')
            if not os.path.exists(opt_out_path):
                os.makedirs(opt_out_path)   
                
            print(b, p)
            gendata(arg, label_out_path, images_out_path, opt_out_path, benchmark=b, part=p)

