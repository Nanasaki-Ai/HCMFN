import os
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DataSetReader(Dataset):
    def __init__(self, args, mode='train'):
        self.exp_dir = args.exp_dir
        if mode == 'train':
            self.label_path = os.path.join(args.data_path, args.dataset, args.dataset_type, 'train_label.pkl')
        else:
            self.label_path = os.path.join(args.data_path, args.dataset, args.dataset_type, 'val_label.pkl')
        self.rgb_path_ntu60 = args.rgb_images_path
        self.suffix = args.image_suffix

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.load_data()


    def load_data(self):
        # label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        strr = 'Load {} samples from {}'.format(len(self.label), self.label_path)
        print(strr)
        with open('{}/log.txt'.format(self.exp_dir), 'a') as f:
            print(strr, file=f)

    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):
        label = self.label[index]
        sample_name_length = len(self.sample_name[index])
        filename = self.sample_name[index][sample_name_length-29:sample_name_length-9]
        rgb = filename + self.suffix
        rgb = np.load(self.rgb_path_ntu60 + rgb)
        rgb = rgb[..., np.newaxis]
        rgb = torch.from_numpy(rgb).float()
        rgb = rgb.permute(2, 0, 1).contiguous()

        return rgb, label