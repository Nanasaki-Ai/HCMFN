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
        # 数据
        if mode == 'train':
            self.label_path = os.path.join(args.data_path, args.dataset, args.dataset_type, 'train_label.pkl')
        else:
            self.label_path = os.path.join(args.data_path, args.dataset, args.dataset_type, 'val_label.pkl')
        self.rgb_path_ntu60 = args.rgb_images_path
        self.suffix = args.image_suffix

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=224),
            # transforms.Resize(size=(225, 45 * self.temporal_rgb_frames)),  #TODO:应该弄成多少呢？224还是225
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        rgb = Image.open(self.rgb_path_ntu60 + rgb)
        width, height = rgb.size
        rgb = np.array(rgb.getdata())
        rgb = torch.from_numpy(rgb).float()
        T, C = rgb.size()
        rgb = rgb.permute(1, 0).contiguous()
        rgb = rgb.view(C, height, width)
        rgb = self.transform(rgb) # resize to 224x224

        return rgb, label