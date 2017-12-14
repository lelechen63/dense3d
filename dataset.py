import os
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader




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

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        _file = open(os.path.join(dataset_dir, "data.pkl"), "rb")
        self.data = pickle.load(_file)
        _file.close()
        print len(self.data)

        self.trainset = []
        for inx in range(len(self.data) - 1):
            for person in self.data[inx].keys():

                self.trainset += self.data[inx][person]
              
        random.shuffle(self.trainset)

        self.testset = []
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train:
            while True:
                # try:

                center = self.trainset[index][1:]
                image = np.load(self.trainset[index][0])
                print image.shape
                ###################################
                # Brats17_TCIA_430_1_flair.nii.gz #
                # Brats17_TCIA_430_1_seg.nii.gz   #
                # Brats17_TCIA_430_1_t1.nii.gz    #
                # Brats17_TCIA_430_1_t1ce.nii.gz  #
                # Brats17_TCIA_430_1_t2.nii.gz    #
                ###################################
                patch = get_patch(image,center,33,33,33)
                musk = get_musk(image,center,9,9,9)
                return patch, musk

                # except:
                #     index = (index + 1) % len(self.train_data)
                #     print 'Fuck'

        else:
            pass

    def __len__(self):
        if self.train:
            return len(self.trainset)
        else:
            return len(self.testset)


    def get_patch(image, center, hsize, wsize, csize):
        """

        :param data: 4D nparray (5,h, w, c)
        :param centers:
        :param hsize:
        :param wsize:
        :param csize:
        :return:
        """
        h, w, c = center[0], center[1], center[2]
        h_beg, w_beg, c_beg = np.maximum(0, h - hsize / 2), np.maximum(0, w - wsize / 2), np.maximum(0,c - csize / 2)
        vox = image[:, h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize]
        return vox

    def get_musk(image, center, hsize, wsize, csize):
        """

        :param data: 4D nparray (5,h, w, c)
        :param centers:
        :param hsize:
        :param wsize:
        :param csize:
        :return:
        """
        h, w, c = center[0], center[1], center[2]
        h_beg, w_beg, c_beg = np.maximum(0, h - hsize / 2), np.maximum(0, w - wsize / 2), np.maximum(0, c - csize / 2)
        vox = image[1, h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize]
        return vox


dataset = BRATSDATA('/mnt/disk1/dat/lchen63/spie', train=True)

data_loader = DataLoader(dataset,
                              batch_size=1,
                              num_workers=4,
                              shuffle=True, drop_last=True)
data_iter = iter(data_loader)
data_iter.next()
print len(data_loader)