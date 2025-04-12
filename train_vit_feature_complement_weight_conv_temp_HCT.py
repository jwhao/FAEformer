import torch
from torch import nn
from data.datamgr import SimpleDataManager , SetDataManager
# from models.predesigned_modules import resnet12
import sys
import os
from utils import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from models.util.pos_embed import interpolate_pos_embed
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# fix seed
np.random.seed(1)
torch.manual_seed(1)
import tqdm
from torch.nn.parallel import DataParallel
# torch.backends.cudnn.benchmark = True
# from models.models_mae import mae_vit_base_patch16,mae_vit_large_patch16
# from sklearn import svm     #导入算法模块
import timm
from torch.utils.tensorboard import SummaryWriter   
# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from models import models_vit_fsl_plus_pos_local_conv
from models import models_vit_fsl_plus_pos_local_conv_support_proto_temp
from models import models_vit_fsl_plus_pos_local_conv_support_proto_hct

import scipy as sp
import scipy.stats
#--------------参数设置--------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=224, type=int, choices=[32, 84, 112, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet',
                                                                   'cub','fs','fc100'])
parser.add_argument('--data_path', default='/data/jiangweihao/data/mini-imagenet',type=str, help='dataset path')
parser.add_argument('--ckp_path', default='/data/jiangweihao/code/HCTransformers/pth/checkpoint_mini.pth',type=str, 
                    help='checkpoint path')

parser.add_argument('--train_n_episode', default=100, type=int, help='number of episodes in meta train')
parser.add_argument('--val_n_episode', default=500, type=int, help='number of episodes in meta val')
parser.add_argument('--test_n_episode', default=2000, type=int, help='number of episodes in meta val')
parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=4, type=int, help='number of unlabeled data in each class')
parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')
parser.add_argument('--model', default='vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')           # 'vit_large_patch16'
parser.add_argument('--global_pool', action='store_true')
parser.add_argument('--batch_size', default=128, type=int, help='total number of batch_size in pretrain')
parser.add_argument('--print_freq', default=100, type=int, help='total number of inner frequency')

parser.add_argument('--momentum', default=0.9, type=float, help='parameter of optimization')
parser.add_argument('--weight_decay', default=5.e-4, type=float, help='parameter of optimization')
parser.add_argument('--lr', default=1e-5, type=float, help='parameter of optimization')
parser.add_argument('--min-lr', default=1e-6, type=float, help='parameter of optimization')

parser.add_argument('--gpu', default='3')
parser.add_argument('--epochs', default=100,type=int)
parser.add_argument('--k', default=98,type=int)    # 每个图像选择的图像块个数
parser.add_argument('--temp', default=10,type=int) 
parser.add_argument('--metric', default='cos',type=str) 
parser.add_argument('--conv-type', default='pool',type=str,choices=['k2','k3','k4','k5','k7','k3d3','k3d2','kds','rd',
                                                                  'pool','linear','attn','mlp']) 

parser.add_argument('--ft', action='store_true')
parser.add_argument('--name', default='7layerC')
parser.add_argument('--layers', default=7, type=int)
parser.add_argument('--drop-path', default=0.2, type=float)
parser.add_argument('--eps', default=0.1, type=float)
parser.add_argument('--weights', default=2, type=int)

params = parser.parse_args()
params.ft = True                               # only for debug
# 设置日志记录路径
log_path = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(log_path,'save_train_k{}_weight_conv_temp-HCT/{}/shot-{}-{}-W{}-{}-{}'.format(
                            params.k,params.dataset,params.n_shot,params.name,params.weights,params.conv_type,time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())))
ensure_path(log_path)
set_log_path(log_path)
log('log and pth save path:  %s'%(log_path))
# log(params)

# -------------设置GPU--------------------
set_gpu(params.gpu)
# -------------导入数据--------------------

json_file_read = False
if params.dataset == 'mini_imagenet':
    base_file = 'train'
    val_file = 'val'
    test_file = 'test'
    params.num_classes = 64
elif params.dataset == 'cub':
    base_file = 'base.json'
    val_file = 'val.json'
    json_file_read = True
    params.num_classes = 200
    params.data_path = '/data/jiangweihao/data/CUB_200_2011'
elif params.dataset == 'tiered_imagenet':
    base_file = 'train'
    val_file = 'val'
    test_file = 'test'
    params.num_classes = 351
    params.data_path = '/data/jiangweihao/data/tiered_imagenet'
elif params.dataset == 'fs':
    base_file = 'train'
    val_file = 'val'
    test_file = 'test'
    params.num_classes = 60
    params.data_path = '/data/jiangweihao/data/CIFAR_FS'
elif params.dataset == 'fc100':
    base_file = 'train'
    val_file = 'val'
    test_file = 'test'
    params.num_classes = 64
    params.data_path = '/data/jiangweihao/data/FC100'
elif params.dataset == 'cod':
    base_file = 'train'
    val_file = 'val'
    test_file = 'test'
    params.num_classes = 48
    # params.image_size = 112
    params.data_path = '/data/jiangweihao/data/COD10K_2'
else:
    ValueError('dataset error')
log(params)
# -----------  base data ----------------------
base_datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size, json_read=json_file_read)
base_loader = base_datamgr.get_data_loader(base_file, aug=True)

#-----------  train data ----------------------
train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
train_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.train_n_episode, json_read=json_file_read, **train_few_shot_params)
train_loader = train_datamgr.get_data_loader(base_file, aug=True)

#------------ val data ------------------------
test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
val_loader = val_datamgr.get_data_loader(val_file, aug=False)
#------------ test data ------------------------
test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
test_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, json_read=json_file_read, **test_few_shot_params)
test_loader = test_datamgr.get_data_loader(test_file, aug=False)
#   ------查看导入的数据----------

# ----------- 导入模型 -------------------------
# # from torchinfo import summary
# # summary(model,[5,3,224,224])

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)

def mean_confidence_interval(data, confidence=0.95):
	a = [1.0*np.array(data[i].cpu()) for i in range(len(data))]
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
	return m,h
       
def train(train_loader,params,model,optimizer,loss_fn,epoch_index):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for episode_index, (temp2,target) in enumerate(train_loader):   
        
        # Measure data loading time
        data_time.update(time.time() - end)

        n,k,c,h,w = temp2.shape
        
        label = np.repeat(range(params.val_n_way),params.n_query)
        label = torch.from_numpy(np.array(label))
        label = label.cuda()

        label_ = np.repeat(range(params.val_n_way),params.n_query*params.val_n_way)
        label_ = torch.from_numpy(np.array(label_))
        label_ = label_.cuda()
        
        imags = temp2.reshape(-1,c,h,w).cuda()
        outputs, outputs_ = model(imags)
       
        
        #----------------------------------------------------soft label--------------------#
        eps = 0.1
        one_hot = torch.zeros_like(outputs).scatter(1, label.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (params.train_n_way - 1)
        log_prb = F.log_softmax(outputs, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.mean()
        # loss = loss_fn(outputs,label)
        #----------------------------------------------------soft label--------------------#
        eps = params.eps
        one_hot = torch.zeros_like(outputs_).scatter(1, label_.view(-1, 1), 1)            #2025-03-15 new add ,之前是label
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (params.train_n_way - 1)
        log_prb = F.log_softmax(outputs_, dim=1)

        loss_ = -(one_hot * log_prb).sum(dim=1)
        loss_ = loss_.mean()
        loss = loss + loss_
        #----------------------------------------------------soft label--------------------#-
        #----------------------------------------------------soft label--------------------#

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        pred = outputs.data.max(1)[1]
        y = np.repeat(range(params.val_n_way),params.n_query)
        y = torch.from_numpy(y)
        y = y.cuda()
        num = params.val_n_way*params.n_query
        pred = pred.eq(y).sum()/num

        losses.update(loss.item(), label.shape[0])
        top1.update(pred, num)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        #============== print the intermediate results ==============#
        if (episode_index+1) % params.print_freq == 0 and episode_index != 0:

            log('Eposide-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(
					epoch_index, episode_index+1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))


    return losses.avg, top1.avg
    
def validate(val_loader,params,model,epoch_index,best_prec1,loss_fn):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
  

	# switch to evaluate mode
    model.eval()
    accuracies = []


    end = time.time()
    for episode_index, (temp2,target) in enumerate(val_loader):   
    
        n,k,c,h,w = temp2.shape
        
        label = np.repeat(range(params.val_n_way),params.n_query)
        label = torch.from_numpy(np.array(label))
        label = label.cuda()

        imags = temp2.reshape(-1,c,h,w).cuda()
        with torch.no_grad():
            outputs, _ = model(imags)
        
        loss = loss_fn(outputs,label)

        pred = outputs.data.max(1)[1]
        y = np.repeat(range(params.val_n_way),params.n_query)
        y = torch.from_numpy(y)
        y = y.cuda()

        num = params.val_n_way*params.n_query
        pred = pred.eq(y).sum()/num
        losses.update(loss.item(), label.size(0))
        top1.update(pred, num)
        accuracies.append(pred)


		# measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        best_prec1 = max(best_prec1,top1.val)
        #============== print the intermediate results ==============#
        if (episode_index+1) % params.print_freq == 0 and episode_index != 0:

            log('Test-({0}): [{1}/{2}]\t' 
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(
					epoch_index, episode_index+1, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
	
        # log(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1))

    return top1.avg, accuracies

    


def main():

    model = models_vit_fsl_plus_pos_local_conv_support_proto_temp.__dict__[params.model](
        num_classes=0,
        global_pool=params.global_pool,
        params = params,
        img_size = params.image_size,
        # attn_drop_rate = 0.2,
        drop_path_rate = params.drop_path #0.2
    )           #         patch_size=8

    
    # chkpt = torch.load('/data/jiangweihao/code/FewTURE/miniImageNet-vit-small-checkpoint1600.pth')
    chkpt = torch.load(params.ckp_path)
    chkpt_state_dict = chkpt['student']
    chkpt_state_dict = match_statedict(chkpt_state_dict)
    # interpolate position embedding
    interpolate_pos_embed(model, chkpt_state_dict)
    msg = model.load_state_dict(chkpt_state_dict, strict=False)
    print(msg)


    # freeze all but the head
    parameters = []
    parameters_ = []
    for _, p in model.named_parameters():
        p.requires_grad = False
        if 'pos_embed' not in _ and 'pos' in _ :              # 当前如果拼接所有特征，需要训练新的位置编码   and 'pos_' not in _ 
            log(_)
            p.requires_grad = True
            parameters.append(p)
        if 'C' in params.name and 'cls' in _:
            log(_)
            p.requires_grad = True
            parameters.append(p)
        if 'weights' in _:
            log(_)
            p.requires_grad = True
            parameters_.append(p)
        if 'conv1d' in _:
            log(_)
            p.requires_grad = True
            parameters_.append(p)

        if params.layers == 12:
            p.requires_grad = True

        for i in range(params.layers):              
            if params.ft and 'blocks.{}'.format(11-i) in _:
                log(_)
                p.requires_grad = True
                parameters.append(p)
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log('number of params:{}'.format(n_parameters))
    model.to('cuda')
    
    # ---------------------------------------------
    loss_fn = torch.nn.CrossEntropyLoss()


    # parameters = [p for p in model.parameters() if p.requires_grad]
    # parameter = [
    #         {'params': parameters, 'lr': params.lr}]  # ,{'params': model.module.head.parameters(), 'lr': 1e-2}
    # optimizer = torch.optim.AdamW(parameter, lr=params.lr)
    # schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = params.epochs, eta_min = params.min_lr)     # eta_min = 1e-4  新增 T_max =  50
    
    parameter = [
             {'params': parameters, 'lr': params.lr}, {'params': parameters_, 'lr': 1e-3}]
    #  ----------新的训练策略-----------
    # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=params.lr, weight_decay=0.001)
    optimizer = torch.optim.Adam(parameter, lr=params.lr, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    log('==========start training ===============')
    
    loss_all = []
    pred_all = []
    best_prec1 = 0
    writer = SummaryWriter('./log/{}/{}-shot/{}'.format(params.dataset,params.n_shot,time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())))
    for epoch in range(params.epochs): 
        log('==========training on train set===============')
        
        epoch_learning_rate1 = schedule.get_lr()
            
        log( 'Train Epoch: {}\tLearning Rate: {}'.format(epoch, epoch_learning_rate1))
        loss,pred = train(train_loader,params,model,optimizer,loss_fn,epoch)

        loss_all.append(loss)
        pred_all.append(pred)

        schedule.step()
        writer.add_scalar('train loss', loss, epoch)
        writer.add_scalar('train acc', pred, epoch)
        writer.add_scalar('learning rate', epoch_learning_rate1[0], epoch)
        
        if epoch >= 0:
            log('============ Validation on the val set ============')
            prec1, _ = validate(val_loader,params,model,epoch,best_prec1,loss_fn)
        
            writer.add_scalar('val acc', prec1, epoch)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # save the checkpoint
            if is_best:
                save_checkpoint(
                    {
                        'epoch_index': epoch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, os.path.join(log_path, 'model_best.pth.tar'))
                log('Best train Epoch: {}\t max accuracy: {}'.format(
                                epoch, best_prec1))

#------------------------------------------------------------------------------------
    log('==========start testing ===============')            
    filename = os.path.join(log_path, 'model_best.pth.tar')
    checkpoint = torch.load(filename)
    
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint['state_dict'])

    model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    # ---------------------------------------------
    loss_fn = torch.nn.CrossEntropyLoss()
    best_prec1 = 0  
    prec1, accuracies = validate(test_loader,params,model,epoch,best_prec1,loss_fn)

    acc, h = mean_confidence_interval(accuracies)
    log('prec1:{:.4f}'.format(prec1))
    log('vag acc:{:.4f}'.format(acc))
    log('confidence_interval:{:.4f}:'.format(h))

if __name__ == '__main__':

    start = time.time()
    main()
    log(time.time()-start)
    log('===========================training end!===================================')