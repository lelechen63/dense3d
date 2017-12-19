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
import os.path as osp
from dataset import  BRATSDATA 
from network import HieNet
from tensorboard_logger import configure, log_value
# from utils import AverageMeter
import torch.nn.functional as F
import datetime
import pytz


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


class Trainer():
    def __init__(self, config):
        self.network = HieNet(config.cuda)

        # self.AverageMeter = AverageMeter()
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

    def get_musk(self, ed , et, net ):
       
        vox = np.copy(net)

       

        vox[np.where(ed[:,:, :, :] != 0)] = 2

        vox[np.where( et[:,:, :, :] != 0)] = 4  

        return  vox


    def fit(self):
        config = self.config
        configure("{}".format(config.log_dir), flush_secs=5)

        num_steps_per_epoch = len(self.data_loader)
        cc = 0
        

        for epoch in range(self.start_epoch, config.max_epochs):
            self.adjust_learning_rate(self.opt,epoch)
            self.network.train()
            for step, (patch, ed, et, net) in enumerate(self.data_loader):
                
                patch, ed, et, net = patch.float(), ed.float(), et.float(), net.float()#, torch.FloatTensor(ed), torch.FloatTensor(et), torch.FloatTensor(net)
                timestamp_start = \
                    datetime.datetime.now(pytz.timezone('US/Eastern'))


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

                loss /= len(patch)
                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while training')
                loss.backward()
                self.opt.step()
                self.network.zero_grad()

                iteration = step + epoch * len(self.data_loader)


                metrics = []
                ed_lbl_pred = f_ed.data.max(1)[1].cpu().numpy()[:, :, :]
                ed_lbl_true = ed.data.cpu().numpy()

                et_lbl_pred = f_et.data.max(1)[1].cpu().numpy()[:, :, :]
                et_lbl_true = et.data.cpu().numpy()
                net_lbl_pred = f_net.data.max(1)[1].cpu().numpy()[:, :, :]
                net_lbl_true = net.data.cpu().numpy()

                lbl_true = self.get_musk(ed_lbl_true,et_lbl_true,net_lbl_true)
                lbl_pred = self.get_musk(ed_lbl_pred,et_lbl_pred,net_lbl_pred)


                for lt, lp in zip(lbl_true, lbl_pred):
                    acc, acc_cls, mean_iu, fwavacc = \
                        label_accuracy_score(
                            [lt], [lp], n_class=5)
                    metrics.append((acc, acc_cls, mean_iu, fwavacc))
                metrics = np.mean(metrics, axis=0)

                with open(osp.join(config.log_dir, 'log.csv'), 'a') as f:
                    elapsed_time = (
                        datetime.datetime.now(pytz.timezone('US/Eastern')) -
                        self.timestamp_start).total_seconds()
                    log = [epoch, iteration] + [loss.data[0]] + \
                        metrics.tolist() + [''] * 5 + [elapsed_time]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')

                    
            if epoch % 1 == 0:
                self.network.eval()
                loss = 0
                ed_acc = 0
                et_acc = 0
                net_acc= 0
                ed_loss_average = 0
                et_loss_average = 0
                net_loss_average = 0
                for step, (patch, ed, et, net) in enumerate(self.evaluation_loader):
                    self.timestamp_start = \
                        datetime.datetime.now(pytz.timezone('US/Eastern'))
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

                
                    ed_loss  = self.bce_loss_fn(f_ed,ed)
                    et_loss  = self.bce_loss_fn(f_et,et)
                    net_loss = self.bce_loss_fn(f_net,net)
                    if np.isnan(float(ed_loss.data[0])) or np.isnan(float(et_loss.data[0])) or np.isnan(float(net_loss.data[0])):
                        raise ValueError('loss is nan while validating')
                    loss += float(ed_loss.data[0]) / len(patch) + float(et_loss.data[0]) / len(patch) + float(net_loss.data[0]) / len(patch)
                    
                


                    metrics = []
                    ed_lbl_pred = f_ed.data.max(1)[1].cpu().numpy()[:, :, :]
                    ed_lbl_true = ed.data.cpu().numpy()

                    et_lbl_pred = f_et.data.max(1)[1].cpu().numpy()[:, :, :]
                    et_lbl_true = et.data.cpu().numpy()
                    net_lbl_pred = f_net.data.max(1)[1].cpu().numpy()[:, :, :]
                    net_lbl_true = net.data.cpu().numpy()

                    lbl_true = self.get_musk(ed_lbl_true,et_lbl_true,net_lbl_true)
                    lbl_pred = self.get_musk(ed_lbl_pred,et_lbl_pred,net_lbl_pred)

                    for lt, lp in zip(lbl_true, lbl_pred):
                        acc, acc_cls, mean_iu, fwavacc = \
                            label_accuracy_score(
                                [lt], [lp], n_class=5)
                        metrics.append((acc, acc_cls, mean_iu, fwavacc))
                    metrics = np.mean(metrics, axis=0)

                    with open(osp.join(config.log_dir, 'validation.csv'), 'a') as f:
                        elapsed_time = (
                            datetime.datetime.now(pytz.timezone('US/Eastern')) -
                            timestamp_start).total_seconds()
                        log = [epoch, iteration] + [loss] + \
                            metrics.tolist() + [''] * 5 + [elapsed_time]
                        log = map(str, log)
                        f.write(','.join(log) + '\n')
                torch.save(self.network,osp.join(self.model_dir,'checkpoint.pth'))







    def adjust_learning_rate(self,optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.lr * (0.1**(epoch // 30))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = self.lr

