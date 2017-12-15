import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


class HieNet(nn.Module):
    
    def __init__(self,cuda_flag):
        super(HieNet,self).__init__()
        self.cuda_flag   = cuda_flag
        dtype            = torch.FloatTensor
        norm_layer		 = nn.BatchNorm3d


        # self.landmark_encoder =nn.Sequential(
        #     nn.ReflectionPad2d(3),
        #     conv2d(1, 64, 7,1, 0),

        #     # conv2d(64,16,3,1,1),
        #     conv2d(64,128,3,2,1),
        #     # conv2d(32,64,3,1,1),
        #     conv2d(128,256,3,2,1),

        #     )
        model = []
        model += [nn.Conv3d(1, 32,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(32),
                      nn.ReLU(True)]
        model += [nn.Conv3d(32, 64,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(64),
                      nn.ReLU(True)]
        model += [nn.Conv3d(64, 128,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(128),
                      nn.ReLU(True)]
        model += [nn.Conv3d(128, 64,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(64),
                      nn.ReLU(True)]
        model += [nn.Conv3d(64, 32,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(32),
                      nn.ReLU(True)]
        model += [nn.Conv3d(32, 16,
                         kernel_size=3, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(16),
                      nn.ReLU(True)]
        
        self.flair_encoder  = nn.Sequential(*model)
        self.t2_encoder 	= nn.Sequential(*model)

        model = []
        model += [nn.Conv3d(2, 32,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(32),
                      nn.ReLU(True)]
        model += [nn.Conv3d(32, 64,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(64),
                      nn.ReLU(True)]
        model += [nn.Conv3d(64, 128,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(128),
                      nn.ReLU(True)]
        model += [nn.Conv3d(128, 64,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(64),
                      nn.ReLU(True)]
        model += [nn.Conv3d(64, 32,
                         kernel_size=5, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(32),
                      nn.ReLU(True)]
        model += [nn.Conv3d(32, 16,
                         kernel_size=3, stride=1,
                         padding=0,
                         bias=True),
                      norm_layer(16),
                      nn.ReLU(True)]


        self.t1_encoder 	= nn.Sequential(*model)



        model = []
        model += [nn.Conv3d(16, 1,
                         kernel_size=3, stride=1,
                         padding=0,
                         bias=True),
                      # norm_layer(2),
                      nn.Sigmoid()]
        self.flair_classifier = nn.Sequential(*model)
        #######output tumor

        model = []
        model += [nn.Conv3d(16 * 2, 1,
                         kernel_size=3, stride=1,
                         padding=0,
                         bias=True),
                      # norm_layer(2),
                      nn.Sigmoid()]
        self.t2_classifier = nn.Sequential(*model)
        ######output core

        model = []
        model += [nn.Conv3d(16 * 3, 1,
                         kernel_size=3, stride=1,
                         padding=0,
                         bias=True),
                      # norm_layer(3),
                      nn.Sigmoid()]
        self.t1_classifier = nn.Sequential(*model)
        ####output enhancing 






    def forward(self, image):

    	flair =  Variable(torch.FloatTensor(image.size(0),1,image.size(2),image.size(3),image.size(4))).cuda() 
    	t1 =  Variable(torch.FloatTensor(image.size(0),2,image.size(2),image.size(3),image.size(4))).cuda() 
    	t2 =  Variable(torch.FloatTensor(image.size(0),1,image.size(2),image.size(3),image.size(4))).cuda() 
    	flair[:,0,:,:,:] = image[:,0,:,:,:]
    	t1[:,:,:,:,:] = image[:,2:4,:,:,:]
    	t2[:,0,:,:,:] = image[:,4,:,:,:]

        t1_feature = self.t1_encoder(t1)
        t2_feature = self.t2_encoder(t2)
        flair_feature = self.flair_encoder(flair)


        edema = self.flair_classifier(flair_feature)


        t2_feature = torch.cat([t2_feature,flair_feature],1)


        enhancing_tumor = self.t2_classifier(t2_feature)

        t1_feature = torch.cat([t1_feature,t2_feature],1)

        none_enhancing = self.t1_classifier(t1_feature)


        return edema, enhancing_tumor, none_enhancing



# a = torch.Tensor(1,5,33,33,33)
# a = Variable(a).cuda()
# net = HieNet(True)
# net = net.cuda()
# print net
# g,gg,ggg = net(a)
# print g.size()
