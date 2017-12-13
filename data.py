import argparse
import os
import numpy as np
from nibabel import load as load_nii
import matplotlib.pyplot as plt
import random
import pickle
import shutil
import time
import multiprocessing

hsize, wsize, csize = 25, 25, 25

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', dest='root', default='/mnt/disk1/dat/lchen63/spie')
    # parser.add_argument('--root', dest='root', default='/media/yue/Data/spie')

    parser.add_argument('--normalization', dest='normalization', type=bool, default=False)
    return parser.parse_args()


def normalization(image):
    # image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()


def read():
    folder = os.path.join(config.root, 'folders')
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(os.path.join(config.root, 'patch')):
        os.mkdir(os.path.join(config.root, 'patch'))

    for i in range(1, 6):
        t = os.path.join(folder, 'folder{}'.format(i))
        if not os.path.exists(t):
            os.mkdir(t)

    gg = os.listdir(os.path.join(config.root, 'Brats17TrainingData'))
    HGG = []
    LGG = []
    for g in gg:
        # print g
        patients = os.listdir(os.path.join(os.path.join(config.root, 'Brats17TrainingData'), g))
        for patient in patients:
            # print patient
            if g == 'HGG':
                HGG.append(os.path.join(os.path.join(config.root, 'Brats17TrainingData'), g, patient))
            else:
                LGG.append(os.path.join(os.path.join(config.root, 'Brats17TrainingData'), g, patient))
    persons = HGG + LGG
    # random.shuffle(persons)
    # print persons
    ff = []
    for i in range(5):
        temp = []
        for k in range(i * len(persons) / 5, (i + 1) * len(persons) / 5):
            temp_path = os.path.join(os.path.join(folder, 'folder{}'.format(i + 1)), persons[k].split('/')[-1])
            temp.append(temp_path)
            if os.path.exists(temp_path):
                continue
            shutil.copytree(persons[k],
                            temp_path)
        # if k % 10 == 0:
        # 	break
        ff.append(temp)
    # break
    data = []
    for f_i in range(5):

        path = os.path.join(os.path.join(config.root, 'patch'), 'folder{}'.format(f_i + 1))
        if not os.path.exists(path):
            os.mkdir(path)

        t = time.time()
        data_t = {}
        data_t['positive'] = []
        data_t['negtive'] = []
        # images = np.ndarray(shape=(len(ff[f_i]), 5, 240, 240, 155))
        images = np.zeros([5, 240, 240, 155])
        positive, negative, whole = [], [], []
        neg, pos, n = 0, 0, 0
        for p, person in enumerate(ff[f_i]):
            print (p, person)
            data = os.listdir(person)
            for i in range(5):
                images[i, :, :, :] = load_nii(os.path.join(person, data[i])).get_data()
            # Brats17_TCIA_430_1_flair.nii.gz
            # Brats17_TCIA_430_1_seg.nii.gz
            # Brats17_TCIA_430_1_t1.nii.gz
            # Brats17_TCIA_430_1_t1ce.nii.gz
            # Brats17_TCIA_430_1_t2.nii.gz
            non_zero_coordinates = np.nonzero(images[1, :, :, :])
            pos += len(non_zero_coordinates[0])

            for inx in range(non_zero_coordinates[0].shape[0]):
                positive.append(
                    (non_zero_coordinates[0][inx], non_zero_coordinates[1][inx], non_zero_coordinates[2][inx]))

            negtive_coordinates = np.where((images[0, :, :, :] != 0) & (images[1, :, :, :] == 0))
            neg += len(negtive_coordinates[0])

            for inx in range(len(negtive_coordinates[0])):
                negative.append(
                    (negtive_coordinates[0][inx], negtive_coordinates[1][inx], negtive_coordinates[2][inx]))

            # head = np.nonzero(images[p, 0, :, :, :])
            # w += head[0].shape[0]
            # for inx in range(head[0].shape[0]):
            #     whole.append((p, head[0][inx], head[1][inx], head[2][inx]))
            patch_path = os.path.join(path, person.split('/')[-1])
            patch_path_pos = os.path.join(patch_path, 'pos')
            patch_path_neg = os.path.join(patch_path, 'neg')
            if not os.path.exists(patch_path):
                os.makedirs(patch_path)
                os.makedirs(patch_path_pos)
                os.makedirs(patch_path_neg)

            print "negtive:{},positive:{}".format(len(negative), len(positive))
            print "n:{},p:{}".format(neg, pos)

            # pos_patches = get_patches_3d(images, positive, 25, 25, 25)
            # neg_patches = get_patches_3d(images, negative, 25, 25, 25)

            # save positive patches
            print 'saving positive patches...'

            pos_process = multiprocessing.Process(target=save_patches_3d,
                                                  args=(images, positive, 25, 25, 25, patch_path_pos))
            pos_process.start()
            # save negative patches
            print 'saving negative patches...'

            neg_process = multiprocessing.Process(target=save_patches_3d,
                                                  args=(images, negative, 25, 25, 25, patch_path_neg))
            neg_process.start()
            neg_process.join()

    print 'done'

        # break
    #     pos_p_p = os.path.join(path, 'positive')
    #     if not os.path.exists(p_p):
    #         os.mkdir(p_p)
    #
        # process = multiprocessing.Process(target=patch_data, args=(positive, pos_p_p, ff[f_i], images,))
        # process.start()
    #
    #     neg_p_p = os.path.join(path, 'negtive')
    #
    #     if not os.path.exists(p_p):
    #         os.mkdir(p_p)
    #
    #     process = multiprocessing.Process(target=patch_data, args=(negtive, neg_p_p, ff[f_i], images,))
    #     process.start()
    #
    #     print 'Folder:	{}	time:	{}'.format(f_i, time.time() - t)
    #
    # with open(os.path.join(config.root, 'data.pkl'), 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def patch_data(sets, save_path, ff_fi, images):
#     for center in sets:
#
#         patch = images[center[0], :, max(center[1] - 12, 0):min(center[1] + 13, images.shape[2]),
#                 max(center[2] - 12, 0):min(center[2] + 13, images.shape[3]),
#                 max(center[3] - 12, 0):min(center[3] + 13, images.shape[4])]
#
#         np.save(save_path, patch)


def save_patches_3d(data, centers, hsize, wsize, csize, patch_path):
    """

    :param data: 4D nparray (h, w, c, ?)
    :param centers:
    :param hsize:
    :param wsize:
    :param csize:
    :return:
    """

    for i in range(len(centers)):
        h, w, c = centers[i][0], centers[i][1], centers[i][2]
        h_beg, w_beg, c_beg = np.maximum(0, h - hsize / 2), np.maximum(0, w - wsize / 2), np.maximum(0,
                                                                                                     c - csize / 2)
        vox = data[:, h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize]
        np.save(os.path.join(patch_path, str(i)), vox)



config = parse_args()
read()