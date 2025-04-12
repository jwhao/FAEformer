"""
加载数据，按照元学习episode模式生成输入数据
一个episode有train_n_episode个batch
img:     torch.Size([5, 21, 3, 84, 84])
label:   torch.Size([5, 21])

数据存放格式支持：ImageFolder，JSON格式
还有一种CSV格式
"""

from datamgr import SetDataManager
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--image_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet','tiered_imagenet','cub'])
parser.add_argument('--data_path', default='/home/jiangweihao/CodeLab/data/mini-imagenet',type=str, help='dataset path')

parser.add_argument('--train_n_episode', default=600, type=int, help='number of episodes in meta train')
parser.add_argument('--val_n_episode', default=300, type=int, help='number of episodes in meta val')
parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=16, type=int, help='number of unlabeled data in each class')
parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')

params = parser.parse_args()

json_file_read = False
if params.dataset == 'mini_imagenet':
        base_file = 'train'
        val_file = 'val'
        params.num_classes = 64
elif params.dataset == 'cub':
    base_file = 'base.json'
    val_file = 'val.json'
    json_file_read = True
    params.num_classes = 200
elif params.dataset == 'tiered_imagenet':
    base_file = 'train'
    val_file = 'val'
    params.num_classes = 351
else:
    ValueError('dataset error')

train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
base_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.train_n_episode, json_read=json_file_read, **train_few_shot_params)
base_loader = base_datamgr.get_data_loader(base_file, aug=True)

target, label = next(iter(base_loader))
print(len(base_loader))
print(target.size())
print(label.size())

# test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
# val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
# val_loader = val_datamgr.get_data_loader(val_file, aug=False)