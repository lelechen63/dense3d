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
    # parser.add_argument('--root', dest='root', default='/media/lele/DATA/spie')

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
    if not os.path.exists(os.path.join(config.root, 'data')):
        os.mkdir(os.path.join(config.root, 'data'))

    for i in range(1, 6):
        t = os.path.join(folder, 'folder{}'.format(i))
        if not os.path.exists(t):
            os.mkdir(t)
        t = os.path.join(os.path.join(config.root, 'data'), 'folder{}'.format(i))
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

        t = time.time()
        data_t = {}
        
       
        for p, person in enumerate(ff[f_i]):
            images = np.zeros([5, 240, 240, 155])
            positive, negative = [], []
            neg, pos, n = 0, 0, 0
            data_t[person.split('/')[-1]] = []
            print (p, person)
            datas = os.listdir(person)
            datas.sort()
            for i in range(5):
                images[i, :, :, :] = load_nii(os.path.join(person, datas[i])).get_data()
                print datas[i]

            ###################################
            # Brats17_TCIA_430_1_flair.nii.gz #
            # Brats17_TCIA_430_1_seg.nii.gz   #
            # Brats17_TCIA_430_1_t1.nii.gz    #
            # Brats17_TCIA_430_1_t1ce.nii.gz  #
            # Brats17_TCIA_430_1_t2.nii.gz    #
            ###################################
            non_zero_coordinates = np.nonzero(images[1, :, :, :])
            pos += len(non_zero_coordinates[0])

            for inx in range(non_zero_coordinates[0].shape[0]):
                positive.append(
                    [non_zero_coordinates[0][inx], non_zero_coordinates[1][inx], non_zero_coordinates[2][inx]])

                # if inx == 100:
                #     break

            negtive_coordinates = np.where((images[0, :, :, :] != 0) & (images[1, :, :, :] == 0))
            neg += len(negtive_coordinates[0])

            for inx in range(len(negtive_coordinates[0])):
                negative.append(
                    [negtive_coordinates[0][inx], negtive_coordinates[1][inx], negtive_coordinates[2][inx]])

                # if inx == 100:
                #     break
            random.shuffle(positive)
            random.shuffle(negative)
            negative = negative[:len(positive)]
            
            data_t[person.split('/')[-1]].append(positive)
            data_t[person.split('/')[-1]].append(negative)
            print len(positive)
            print len(negative)
            numpy_path = person.replace('folders','data') + '.npy'
            np.save(numpy_path,images)
        if len(data_t) != 57:
            print '___________________'
            break
        data.append(data_t)


            # patch_path = os.path.join(path, person.split('/')[-1])
            # patch_path_pos = os.path.join(patch_path, 'pos')
            # patch_path_neg = os.path.join(patch_path, 'neg')
            # if not os.path.exists(patch_path):
            #     os.makedirs(patch_path)
            #     os.makedirs(patch_path_pos)
            #     os.makedirs(patch_path_neg)

            # print "negtive:{},positive:{}".format(len(negative), len(positive))
            # print "n:{},p:{}".format(neg, pos)

           
            # # save positive patches
            # print 'saving positive patches...'

            # pos_process = multiprocessing.Process(target=save_patches_3d,
            #                                       args=(images, positive, 25, 25, 25, patch_path_pos))
            # pos_process.start()
            # # save negative patches
            # print 'saving negative patches...'

            # neg_process = multiprocessing.Process(target=save_patches_3d,
            #                                       args=(images, negative, 25, 25, 25, patch_path_neg))
            # neg_process.start()
            # neg_process.join()

    print 'done'
    with open(os.path.join(config.root,'data.pkl'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
def merge_two_dicts(x, y):
    z = x.copy()   
    z.update(y)    
    return z
def generate_train_data(data):
    _file = open(data, "rb")
    data = pickle.load(_file)
    _file.close()
    print len(data)
    count = 0
    # train = {}
    # for inx in range(len(data)-1):
    #     train = merge_two_dicts(train,data[inx])
    # print len(train)
    train = []
    for inx in range(len(data) - 1):
        for person in data[inx].keys():
            print person
            image = np.load(os.path.join(os.path.join(os.path.join(config.root,'data'),'folder{}'.format(inx+1)),person+'.npy'))
            print image.shape
            
            positive = data[inx][person][0]
            negative = data[inx][person][1]
            # print len(positive)
            # print len(negative)
            # print '----------'
            random.shuffle(positive)
            random.shuffle(negative)
            negative = negative[:len(positive)]

            for center in positive+negative:
                # print center
                patch = save_patches_3d(image,center,33,33,33)
                # print patch.shape
                train.append(patch)
            # break
        # break
    print len(train)
    # print train[0]
    with open(os.path.join(config.root,'train.pkl'), 'wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)

            

    # for i in  data:
    #     print type(i)
    #     print i.keys()
    #     print len(i)
        # count += len(i)
    # print count


    # print data



def save_patches_3d(image, center, hsize, wsize, csize):
    """

    :param data: 4D nparray (5,h, w, c)
    :param centers:
    :param hsize:
    :param wsize:
    :param csize:
    :return:
    """

    # for i in range(len(centers)):
    h, w, c = center[0], center[1], center[2]
    h_beg, w_beg, c_beg = np.maximum(0, h - hsize / 2), np.maximum(0, w - wsize / 2), np.maximum(0,
                                                                                                 c - csize / 2)
    vox = image[:, h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize]
    # np.save(os.path.join(patch_path, str(i)), vox)
    return vox


config = parse_args()
read()
# generate_train_data(os.path.join(config.root, "data.pkl"))