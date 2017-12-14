import os
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image




class BRATSDATA(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[64, 64],
                 train=True):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

       
        _file = open(os.path.join(dataset_dir, "data.pkl"), "rb")
        self.data = pickle.load(_file)
        _file.close()
       

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train:
            while True:
                try:
                    paths = self.data[index]["image_path"]
                    while True:
                        # print paths
                        a = random.randint(0,29)
                        if paths[0][-8] == '_':
                            example_path = paths[0][:-8] +'_%03d.jpg' % (a)
                            if os.path.isfile(example_path):
                                break
                        elif paths[0][-9] == '_':
                            example_path = paths[0][:-9] +'_%03d.jpg' % (a)
                            if os.path.isfile(example_path):
                                break
                    example_lip = cv2.imread(example_path)
                    if example_lip is None:
                        raise IOError
                    example_lip = cv2.cvtColor(example_lip, cv2.COLOR_BGR2RGB)
                    example_lip = cv2.resize(example_lip, self.output_shape)

                    example_lip = self.transform(example_lip)
                    example_landmark = np.load(example_path.replace('lips','landmark2d').replace('jpg','npy'))

                    im_cub = torch.FloatTensor(29,3,self.output_shape[0],self.output_shape[1])
                    for i,path in enumerate(paths):
                        # im = Image.open(path).convert("RGB").resize(self.output_shape)
                        im = cv2.imread(path)
                        if im is None:
                            raise IOError
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = cv2.resize(im, self.output_shape)
                        im = self.transform(im)
                        im_cub[i,:,:,:] = im
                    # im_cub = im_cub.permute(1,0,2,3)
                    wrong_index = random.choice(
                        [x for x in range(self.__len__()) if x != index])

                    right_lmss = self.train_data[index]["lms_path"]
                    right_embed = torch.FloatTensor(29,1,128,4)
                    for i,lms_path in enumerate(right_lmss):
                        right_embed[i,0,:,:] = torch.FloatTensor(np.load(lms_path))

                    landmarks = torch.FloatTensor( 29,1,self.output_shape[0],self.output_shape[1])
                    for i, land_path in enumerate(self.train_data[index]['landmark_path']):
                        landmarks[i,:,:,:] =  torch.FloatTensor(np.load(land_path))

                    wrong_lmss = self.train_data[wrong_index]["lms_path"]
                    wrong_embed = torch.FloatTensor(29,1,128,4)
                    for i,lms_path in enumerate(wrong_lmss):
                        wrong_embed[i,0,:,:] = torch.FloatTensor(np.load(lms_path))
                    wrong_landmarks = torch.FloatTensor(29,1,self.output_shape[0],self.output_shape[1])
                    for i, land_path in enumerate(self.train_data[wrong_index]['landmark_path']):
                        wrong_landmarks[i,:,:,:] = torch.FloatTensor(np.load(land_path))
                    return example_lip, example_landmark, im_cub, landmarks, wrong_landmarks, right_embed, wrong_embed
                except:
                    index = (index + 1) % len(self.train_data)
                    print 'Fuck'

        else:
            while True:
                try:
                    paths = self.test_data[index]["image_path"]
                    while True:
                        a = random.randint(0, 29)
                        if paths[0][-8] == '_':
                            example_path = paths[0][:-8] + '_%03d.jpg' % (a)
                            if os.path.isfile(example_path):
                                break
                        elif paths[0][-9] == '_':
                            example_path = paths[0][:-9] + '_%03d.jpg' % (a)
                            if os.path.isfile(example_path):
                                break
                    example_lip = cv2.imread(example_path)
                    example_landmark = np.load(example_path.replace('lips','landmark2d').replace('jpg','npy'))
                    if example_lip is None:
                        raise IOError
                    example_lip = cv2.cvtColor(example_lip, cv2.COLOR_BGR2RGB)
                    example_lip = cv2.resize(example_lip, self.output_shape)
                    # example_lip = Image.open(example_path).convert("RGB").resize(self.output_shape)
                    example_lip = self.transform(example_lip)

                    im_cub = torch.FloatTensor(29, 3, self.output_shape[0], self.output_shape[1])
                    for i, path in enumerate(paths):
                        # im = Image.open(path).convert("RGB").resize(self.output_shape)
                        im = cv2.imread(path)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = cv2.resize(im, self.output_shape)
                        im = self.transform(im)
                        im_cub[i, :, :, :] = im
                    # im_cub = im_cub.permute(1, 0, 2, 3)
                    right_lmss = self.test_data[index]["lms_path"]
                    right_embed = torch.FloatTensor(29, 1, 128, 4)
                    for i, lms_path in enumerate(right_lmss):
                        right_embed[i, 0,:,  :] = torch.FloatTensor(np.load(lms_path))
                    landmarks = torch.FloatTensor(29, 1, 64 , 64)
                    for i,land_path in enumerate(self.test_data[index]['landmark_path']):
                        landmarks[i,:,:,:] = torch.FloatTensor(np.load(land_path))

            

                    caption = self.test_data[index]["lms_path"]

                    return example_lip, example_landmark, im_cub, landmarks, right_embed, caption
                    # return example_lip,im_cub, right_embed, caption
                except:
                    index = (index + 1) % len(self.test_data)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

