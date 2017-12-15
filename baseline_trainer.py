import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from dataset import  BRATSDATA 
from network import HieNet
from tensorboard_logger import configure, log_value
from utils import AverageMeter
class Trainer():
    def __init__(self, config):
        self.network = HieNet(config.cuda)

        self.AverageMeter = AverageMeter()
        print(self.network)
        self.bce_loss_fn = nn.BCELoss()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.lr = config.lr

        self.opt = torch.optim.Adam(self.network.parameters(), lr=self.lr , weight_decay=1e-4)#, betas=(config.beta1, config.beta2))
       
        
        self.dataset = BRATSDATA(config.dataset_dir, train=config.is_train)
        

        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        data_iter = iter(self.data_loader)
        data_iter.next()


        self.evaluationset = BRATSDATA(config.dataset_dir, train=False)
        

        self.evaluation_loader = DataLoader(self.evaluationset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        evaluation_iter = iter(self.evaluation_loader)
        evaluation_iter.next()

        ########multiple GPU####################
        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            self.network     = nn.DataParallel(self.network.cuda(), device_ids=device_ids)
            self.bce_loss_fn   = self.bce_loss_fn.cuda()
            self.mse_loss_fn   = self.mse_loss_fn.cuda()
        #########single GPU#######################

        # if config.cuda:
        #     device_ids = [int(i) for i in config.device_ids.split(',')]
        #     self.generator     = self.generator.cuda()
        #     self.bce_loss_fn   = self.bce_loss_fn.cuda()
        #     self.mse_loss_fn   = self.mse_loss_fn.cuda()
        #     self.ones          = self.ones.cuda()
        #     self.zeros         = self.zeros.cuda()
        #     self.zeros         = self.zeros.cuda()



        self.config = config
        self.start_epoch = 0

        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)

    def fit(self):
        config = self.config
        configure("{}".format(config.log_dir), flush_secs=5)

        num_steps_per_epoch = len(self.data_loader)
        cc = 0
        

        for epoch in range(self.start_epoch, config.max_epochs):
            self.adjust_learning_rate(self.opt,epoch)
            for step, (patch, ed, et, net) in enumerate(self.data_loader):
                
                patch, ed, et, net = patch.float(), ed.float(), et.float(), net.float()#, torch.FloatTensor(ed), torch.FloatTensor(et), torch.FloatTensor(net)
                t1 = time.time()


                if config.cuda:
                    patch = Variable(patch).cuda()
                    ed = Variable(ed).cuda()
                    et = Variable(et).cuda()
                    net    = Variable(net).cuda()
                    
                else:
                    patch = Variable(patch)
                    ed = Variable(ed)
                    et = Variable(et)
                    net    = Variable(net)


                

                f_ed, f_et, f_net = self.network(patch)
                loss = 0

                ed_loss  = self.bce_loss_fn(f_ed,ed)
                et_loss  = self.bce_loss_fn(f_et,et)
                net_loss = self.bce_loss_fn(f_net,net)

                loss = ed_loss + et_loss + net_loss
                loss.backward()
                self.opt.step()
                self.network.zero_grad()

                t2 = time.time()

                if (step+1) % 1 == 50 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch
                    eta = int((t2-t1)*steps_remain)

                    print("[{}/{}][{}/{}] loss: {:.4f}, ed_loss: {:.4f},et_loss: {:.4f},net_loss: {:.4f}, ETA: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss.data[0], ed_loss.data[0], et_loss.data[0], net_loss.data[0],  eta))
                    log_value('training_loss',loss.data[0] , step + num_steps_per_epoch * epoch)
                if (step ) % (num_steps_per_epoch/1000) == 0 :
                    _ed_iou = self.dice_coef_np(ed.data.cpu().numpy(), f_ed.data.cpu().numpy(),2)
                    _et_iou = self.dice_coef_np(et.data.cpu().numpy(), f_et.data.cpu().numpy(),2)
                    _net_iou = self.dice_coef_np(net.data.cpu().numpy(), f_net.data.cpu().numpy(),2)

                    print("[{}/{}][{}/{}]  ed_bg: {:.4f}, ed_iou: {:.4f}, et_bg: {:.4f},et_iou: {:.4f}, net_bg: {:.4f}, net_iou: {:.4f}"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, _ed_iou[0], _ed_iou[1] , _et_iou[0], _et_iou[1], _net_iou[0], _net_iou[1]))
                if step == 200:
                    break
            if epoch % 1 == 0:
                loss = 0
                ed_acc = 0
                et_acc = 0
                net_acc= 0
                ed_loss_average = 0
                et_loss_average = 0
                net_loss_average = 0
                for step, (patch, ed, et, net) in enumerate(self.evaluation_loader):
                    t1 = time.time()
                    patch, ed, et, net = patch.float(), ed.float(), et.float(), net.float()

                    if config.cuda:
                        patch = Variable(patch).cuda()
                        ed = Variable(ed).cuda()
                        et = Variable(et).cuda()
                        net    = Variable(net).cuda()
                        
                    else:
                        patch = Variable(patch)
                        ed = Variable(ed)
                        et = Variable(et)
                        net    = Variable(net)


                    

                    f_ed, f_et, f_net = self.network(patch)

                    # ed_prec = self.accuracy(f_ed,ed)
                    # et_prec = self.accuracy(f_et,et)
                    # net_prec =self.accuracy(f_net,net)


                    ed_loss  = self.bce_loss_fn(f_ed,ed)
                    et_loss  = self.bce_loss_fn(f_et,et)
                    net_loss = self.bce_loss_fn(f_net,net)

                    loss += ed_loss + et_loss + net_loss
                    ed_loss_average += ed_loss
                    et_loss_average += et_loss
                    net_loss_average += net_loss

                    _ed_iou = self.dice_coef_np(ed.cpu().data.numpy(), f_ed.data.cpu().numpy(),2)
                    _et_iou = self.dice_coef_np(et.cpu().data.numpy(), f_et.data.cpu().numpy(),2)
                    _net_iou = self.dice_coef_np(net.cpu().data.numpy(), f_net.data.cpu().numpy(),2)

                    ed_acc += _ed_iou
                    et_acc += _et_iou
                    net_acc += _net_iou

                loss = loss/step
                ed_acc = ed_acc/step
                et_acc = et_acc/step
                net_acc = net_acc/step


                ed_loss_average = ed_loss_average/step
                et_loss_average = et_loss_average/step
                net_loss_average = net_loss_average/step
                print '==================================Evaluation========================================================'
                print("[{}/{}][{}/{}] loss: {:.4f}, ed_loss: {:.4f},et_loss: {:.4f},net_loss: {:.4f}"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss.data[0], ed_loss.data[0], et_loss.data[0], net_loss.data[0]))
                
                print("[{}/{}][{}/{}]  ed_bg: {:.4f}, ed_iou: {:.4f}, et_bg: {:.4f},et_iou: {:.4f}, net_bg: {:.4f}, net_iou: {:.4f}"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, _ed_iou[0], _ed_iou[1] , _et_iou[0], _et_iou[1], _net_iou[0], _net_iou[1]))

                # log_value('evaluation_loss',loss.data[0] , step + num_steps_per_epoch * epoch)








                torch.save(self.network.state_dict(),
                           "{}/HieNet{}.pth"
                           .format(config.model_dir,epoch))

    # def load(self, directory, epoch):
    #     gen_path = os.path.join(directory, 'generator_{}.pth'.format(epoch))

    #     self.generator.load_state_dict(torch.load(gen_path))

    #     print("Load pretrained [{}, {}]".format(gen_path, disc_path))

    def adjust_learning_rate(self,optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.lr * (0.1**(epoch // 30))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = self.lr


    def dice_coef_np(self, y_true, y_pred, num_classes):
        """

       :param y_true: sparse labels
        :param y_pred: sparse labels
        :param num_classes: number of classes
        :return:
        """
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        y_true = y_true.flatten()
        y_true = self.one_hot(y_true, num_classes)
        y_pred = y_pred.flatten()
        y_pred = self.one_hot(y_pred, num_classes)
        intersection = np.sum(y_true * y_pred, axis=0)
        return (2. * intersection) / (np.sum(y_true, axis=0) + np.sum(y_pred, axis=0))


    def one_hot(self, y, num_classees):
        y_ = np.zeros([len(y), num_classees])
        y_[np.arange(len(y)), y] = 1
        return y_