import argparse
import os
import numpy as np
from nibabel import load as load_nii
import matplotlib.pyplot as plt
import random
import pickle
import shutil
import time
def parse_args():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--root', dest='root', default='/media/lele/DATA/spie')
	parser.add_argument('--root', dest='root', default='/mnt/disk1/dat/lchen63/spie')
	
	parser.add_argument( '--normalization', dest='normalization', type=bool, default=False)
	return parser.parse_args()


def normalization(image):
	
	# image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()

def read():

	folder = os.path.join(config.root,'folders')
	if not os.path.exists(folder):
		os.mkdir(folder)
	if not os.path.exists(os.path.join(config.root,'patch')):
		os.mkdir(os.path.join(config.root,'patch'))

	for i in range(1,6):
		t = os.path.join(folder,'folder{}'.format(i))
		if not os.path.exists(t):
			os.mkdir(t)

	gg = os.listdir(os.path.join(config.root,'Brats17TrainingData'))
	HGG = []
	LGG = []
	for g in gg:
		# print g
		patients = os.listdir(os.path.join(os.path.join(config.root,'Brats17TrainingData'),g))
		for patient in patients:
			# print patient
			if g == 'HGG':
				HGG.append(os.path.join(os.path.join(config.root,'Brats17TrainingData'),g,patient))
			else:
				LGG.append(os.path.join(os.path.join(config.root,'Brats17TrainingData'),g,patient))
	persons = HGG + LGG
	# random.shuffle(persons)
	# print persons
	ff = []
	for i in range(5):
		temp = []

		for k in range(i*len(persons)/5,(i+1)*len(persons)/5):
			shutil.copytree(persons[k], os.path.join(os.path.join(folder,'folder{}'.format(i+1)),persons[k].split('/')[-1]))
			temp.append(os.path.join(os.path.join(folder,'folder{}'.format(i+1)),persons[k].split('/')[-1]))
			# if k % 10 == 0:
			# 	break
		ff.append(temp)
		# break
	data = []
	for f_i in range(5):

		path = os.path.join(os.path.join(config.root,'patch'),'folder{}'.format(f_i+1))
		if not os.path.exists(path):
			os.mkdir(path)

		t = time.time()
		data_t = {}
		data_t['positive'] = []
		data_t['negtive'] = []
		images = np.ndarray(shape=(len(ff[f_i]),5,240,240,155))
		positive = set()
		negtive = set()
		whole = set()
		w = 0
		p = 0
		n = 0
		for p,person in enumerate(ff[f_i]):
			data = os.listdir(person)
			for i in range(5):
				images[p,i,:,:,:] = load_nii(os.path.join(person,data[i])).get_data()
				# Brats17_TCIA_430_1_flair.nii.gz
				# Brats17_TCIA_430_1_seg.nii.gz
				# Brats17_TCIA_430_1_t1.nii.gz
				# Brats17_TCIA_430_1_t1ce.nii.gz
				# Brats17_TCIA_430_1_t2.nii.gz
			non_zero_coordinates= np.nonzero(images[p,1,:,:,:])
			p += non_zero_coordinates[0].shape[0]
			
			for inx in range(non_zero_coordinates[0].shape[0]):
				positive.add((p,non_zero_coordinates[0][inx],non_zero_coordinates[1][inx],non_zero_coordinates[2][inx]))

			head = np.nonzero(images[p,0,:,:,:])
			w += head[0].shape[0]
			for inx in range(head[0].shape[0]):
				whole.add((p,head[0][inx],head[1][inx],head[2][inx]))

		negtive = whole - positive
		print "negtive:{},positive:{}".format(len(negtive),len(positive))
		print "n:{},p:{}".format(w - p ,p)
		break
		pos_p_p = os.path.join(path,'positive')
		if not os.path.exists(p_p):
			os.mkdir(p_p)
		
		
		
		process = multiprocessing.Process(target = patch_data,args = (positive,pos_p_p,ff[f_i],images,))
        process.start()
	
		neg_p_p = os.path.join(path,'negtive')

		if not os.path.exists(p_p):
			os.mkdir(p_p)

		process = multiprocessing.Process(target = patch_data,args = (negtive,neg_p_p,ff[f_i],images,))
        process.start()

		
		print 'Folder:	{}	time:	{}'.format(f_i,time.time()-t)


	with open(os.path.join(config.root,'data.pkl'), 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def patch_data(sets,p_p,ff_fi,images):
	for center in sets:
		if not os.path.exists(os.path.join(p_p,ff_fi[center[0]].split('/')[-1])):
			os.mkdir(os.path.join(p_p,ff_fi[center[0]].split('/')[-1]))
		# print center
		patch = images[center[0],:,max(center[1]-12,0):min(center[1]+13,images.shape[2]),max(center[2]-12,0):min(center[2]+13,images.shape[3]),max(center[3]-12,0):min(center[3]+13,images.shape[4])]
		np.save(os.path.join(os.path.join(p_p,ff_fi[center[0]].split('/')[-1]),'{}_{}_{}.npy'.format(center[1],center[2],center[3])),patch)
			








# def get_balance_center(test,train):
# 	positive = set()
# 	negtive = set()
# 	whole = set()
# 	train_positive_txt = os.path.join(config.folder,'t_p.txt')
# 	train_negtive_txt = os.path.join(config.folder,'t_n.txt')
	



# 	###########################test##################################
# 	path = os.path.join(config.patch_save,'test')
# 	if not os.path.exists(path):
# 		os.mkdir(path)
# 	for p in range(test.shape[0]):
		

# 		non_zero_coordinates= np.nonzero(test[p,1,:,:,:])
		
# 		for inx in range(non_zero_coordinates[0].shape[0]):
# 			positive.add((p,non_zero_coordinates[0][inx],non_zero_coordinates[1][inx],non_zero_coordinates[2][inx]))

# 		head = np.nonzero(test[p,0,:,:,:])
# 		for inx in range(head[0].shape[0]):
# 			whole.add((p,head[0][inx],head[1][inx],head[2][inx]))

	
# 	negtive = whole - positive

# 	p_p = os.path.join(path,'positive')
# 	if not os.path.exists(p_p):
# 		os.mkdir(p_p)
# 	for center in positive:
# 		patch = test[center[0],:,max(center[1]-12,0):min(center[1]+13,test.shape[2]),max(center[2]-12,0):min(center[2]+13,test.shape[3]),max(center[3]-12,0):min(center[3]+13,test.shape[4])]
# 		np.save(os.path.join(p_p,'{}_{}_{}_{}.npy'.format(center[0],center[1],center[2],center[3])),patch)
# 		break
# 	p_p = os.path.join(path,'negtive')
# 	if not os.path.exists(p_p):
# 		os.mkdir(p_p)
# 	for center in negtive:
# 		patch = test[center[0],:,max(center[1]-12,0):min(center[1]+13,test.shape[2]),max(center[2]-12,0):min(center[2]+13,test.shape[3]),max(center[3]-12,0):min(center[3]+13,test.shape[4])]
# 		np.save(os.path.join(p_p,'{}_{}_{}_{}.npy'.format(center[0],center[1],center[2],center[3])),patch)
# 		break
# 	##########################train#########################################
# 	path = os.path.join(config.patch_save,'train')
# 	if not os.path.exists(path):
# 		os.mkdir(path)
# 	for p in range(train.shape[0]):
	

# 		non_zero_coordinates= np.nonzero(train[p,1,:,:,:])
	
# 		for inx in range(non_zero_coordinates[0].shape[0]):
# 			positive.add((p,non_zero_coordinates[0][inx],non_zero_coordinates[1][inx],non_zero_coordinates[2][inx]))

# 		head = np.nonzero(train[p,0,:,:,:])
# 		for inx in range(head[0].shape[0]):
# 			whole.add((p,head[0][inx],head[1][inx],head[2][inx]))

# 	negtive = whole - positive

# 	p_p = os.path.join(path,'positive')
# 	if not os.path.exists(p_p):
# 		os.mkdir(p_p)
# 	for center in positive:
# 		patch = test[center[0],:,max(center[1]-12,0):min(center[1]+13,test.shape[2]),max(center[2]-12,0):min(center[2]+13,test.shape[3]),max(center[3]-12,0):min(center[3]+13,test.shape[4])]
# 		np.save(os.path.join(p_p,'{}_{}_{}_{}.npy'.format(center[0],center[1],center[2],center[3])),patch)
# 		break
# 	p_p = os.path.join(path,'negtive')
# 	if not os.path.exists(p_p):
# 		os.mkdir(p_p)
# 	for center in negtive:
		
# 		patch = test[center[0],:,max(center[1]-12,0):min(center[1]+13,test.shape[2]),max(center[2]-12,0):min(center[2]+13,test.shape[3]),max(center[3]-12,0):min(center[3]+13,test.shape[4])]
# 		np.save(os.path.join(p_p,'{}_{}_{}_{}.npy'.format(center[0],center[1],center[2],center[3])),patch)

# 		break


	# print image.shape
	# for p in range(image.shape[0]):
	# 	for i in range(image.shape[2]):
	# 		for j in range(image.shape[3]):
	# 			for k in range(image.shape[4]):
	# 				if (p,i,j,k) not in positive:
	# 					# flage = 0
	# 					# for i_i in range(max(0,i-16),min(image.shape[2],i+16)):
	# 					# 	for j_j in range(max(0,j-16),min(image.shape[3],j+16)):
	# 					# 		for k_k in range(max(0,k-16),min(image.shape[4],k+16)):  
	# 					# 			if (p,i_i,j_j,k_k) in positive:

	# 					negtive.add((p,i,j,k))
	# 				whole.add((p,i,j,k))

	# print len(whole)
	# print len(negtive)
	# print len(positive)
	# for i in negtive:
	# 	print i
	# 	break
	# for i in positive:
	# 	print i
	# 	break











config = parse_args()
read()
# test,train = read(config.folder)
# get_balance_center(test,train)
# a = set()
# gg = (1,3)
# print type(gg)
# a.add(gg)
# a.add((3,3))
# print a
# b = set()
# b.add(gg)
# print a- b