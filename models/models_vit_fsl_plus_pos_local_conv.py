# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

import torch.nn.functional as F
def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t()) 
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp
def patch_selection(sim,fmap,k):
     # 使用 torch.topk 获取每行前k个最大值的索引
     top_k_values, top_k_indices = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)

     # 生成前k个行的掩码
     top_mask = torch.zeros_like(sim, dtype=torch.bool)
     top_mask.scatter_(1, top_k_indices, True)

     # 应用掩码
     patch_select = fmap * top_mask.unsqueeze(2)

     # 使用 torch.masked_select 过滤出满足条件的元素
     patch_select = torch.masked_select(patch_select, top_mask.unsqueeze(2)).reshape(sim.size(0),k,-1)
     return patch_select

class ResidualDilatedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(384, 384, 3, stride=2, dilation=3, padding=3)
    
    def forward(self, x):
        return x[:,:,::2] + self.conv(x)
    
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,params=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.query = params.n_query
        self.shot = params.n_shot
        self.k = 98
        self.metric = params.metric
        self.type = params.conv_type
        self.temp=params.temp
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.pos = nn.Parameter(torch.zeros(1, 2*(self.k+1), kwargs['embed_dim']))
        self.weights = nn.Parameter(torch.ones(3))  # 初始权重为[1, 1, 1]
        # self.weights = torch.nn.functional.softmax(self.weights, dim=0)
        # self.conv1d = nn.Conv2d(196*2, 196, kernel_size=1)
        if self.type == 'k2':
            self.conv1d = nn.Conv1d(                                               
                in_channels=kwargs['embed_dim'],
                out_channels=kwargs['embed_dim'],  # 保持特征维度不变
                kernel_size=2,  # 可以根据需要调整                     
                stride=2,       # 步幅为2，可以减半序列长度
                padding=0       # 适当的填充可以保持特征维度的稳定性      
            )
        elif self.type == 'k3':
            self.conv1d = nn.Conv1d(                                               
                    in_channels=kwargs['embed_dim'],
                    out_channels=kwargs['embed_dim'],  # 保持特征维度不变
                    kernel_size=3,  # 可以根据需要调整                      #2 
                    stride=2,       # 步幅为2，可以减半序列长度
                    padding=1       # 适当的填充可以保持特征维度的稳定性      # 0
                )
        elif self.type == 'k4':                                                              
            # 增大卷积核
            self.conv1d = nn.Conv1d(
                    in_channels=384,
                    out_channels=384,
                    kernel_size=4,  # 更大的卷积核
                    stride=2,
                    padding=1       # 保持输出长度 (392-4+2*1)/2 +1 = 196
                )
        elif self.type == 'k5':                                                              
            # 增大卷积核
            self.conv1d = nn.Conv1d(
            in_channels=384,
            out_channels=384,
            kernel_size=5,        # 卷积核长度
            stride=2,             # 步长=2实现下采样
            padding=2,            # 填充计算见下文
            padding_mode='zeros'  # 默认填充模式
        )
        elif self.type == 'k6':                                                              
            # 增大卷积核
            self.conv1d = nn.Conv1d(
            in_channels=384,
            out_channels=384,
            kernel_size=6,        # 卷积核长度
            stride=2,             # 步长=2实现下采样
            padding=2,            # 填充计算见下文
            padding_mode='zeros'  # 默认填充模式
        )
        elif self.type == 'k7':                                                              
            # 增大卷积核
            self.conv1d = nn.Conv1d(
            in_channels=384,
            out_channels=384,
            kernel_size=7,
            stride=2,
            padding=3  # 计算方式同上
        )
        elif self.type == 'k3d3':
            # 增大膨胀率（dilation）
            self.conv1d = nn.Conv1d(
                    in_channels=384,
                    out_channels=384,
                    kernel_size=3,
                    stride=2,
                    dilation=3,  # 膨胀率=3，实际感受野=3*2+1=7
                    padding=3     # 保持输出长度 (392-3*2-1 +2*3)/2 +1 = 196
                )
        elif self.type == 'k3d2':
            # 组合使用（大核 + 膨胀）
            self.conv1d = nn.Conv1d(
                    in_channels=384,
                    out_channels=384,
                    kernel_size=3,
                    stride=2,
                    dilation=2,  # 感受野=3*2+1=7
                    padding=2     # 保持输出长度
                )
        elif self.type == 'kds':
            # 多层膨胀卷积：堆叠多个膨胀卷积层（如WaveNet结构），逐层增加膨胀率：
            self.conv1d = nn.Sequential(
                    nn.Conv1d(384, 384, 3, stride=1, dilation=1, padding=1),
                    nn.Conv1d(384, 384, 3, stride=1, dilation=2, padding=2),
                    nn.Conv1d(384, 384, 3, stride=2, dilation=4, padding=4)
                )
        elif self.type == 'rd':
            # 残差膨胀卷积：结合残差连接避免梯度消失
            self.conv1d = ResidualDilatedConv()
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        i = 0
        
        n_query = self.query          # online =5
        n_shot = self.shot
        for blk in self.blocks:
            x = blk(x)                                         # [30 =n_way*(n_shot+n_query),197,768]
            i += 1
            if i == 11:
        
                _,p,d = x.shape
                x = x.reshape(5,-1,p,d)                        # [5,6,197,768]
                xs = x[:,:n_shot].reshape(-1,p,d)    # [25,197,384]
                
                xq = x[:,n_shot:].reshape(-1,p,d)    # [75,197,384]
                
                #---------------------
                # 求解cls与每一个patch的相似度
                query_class = xq[:, 0, :].unsqueeze(1)  # Q x 1 x C
                query_image = xq[:, 1:, :]  # Q x L x C

                support_class = xs[:, 0, :].unsqueeze(1)  # KS x 1 x C
                support_image = xs[:, 1:, :]  # KS x L x C


                # ---------通过一维卷积压缩序列长度   196---> 98------------
                query_image = query_image.permute(0,2,1)
                support_image = support_image.permute(0,2,1)
                query_image_select = self.conv1d(query_image)
                support_image_select = self.conv1d(support_image)
                query_image_select = query_image_select.permute(0,2,1)
                support_image_select = support_image_select.permute(0,2,1)

                xs = torch.cat((support_class,support_image_select),dim=1)
                xq = torch.cat((query_class,query_image_select),dim=1)
                xs = xs.unsqueeze(0).repeat(5*n_query,1,1,1)
                xq = xq.unsqueeze(1).repeat(1,5*n_shot,1,1)
                
                x = torch.cat((xs,xq),dim=2).reshape(-1,(self.k+1)*2,d)
                x = x + self.pos   # ------------------加上新的位置编码
                
        self.k = 98
        #-------------------------获得特征------------------------------
        x = self.norm(x)
        xs = x[:,0]
        xq = x[:,self.k+1]
        xs_patch = x[:,1:self.k+1]
        xq_patch = x[:,self.k+2:]
            
        xs = xs.reshape(5*n_query,-1,d)
        xq = xq.reshape(5*n_query,-1,d) 
        xs_patch = xs_patch.reshape(5*n_query,-1,self.k,d)
        xq_patch = xq_patch.reshape(5*n_query,-1,self.k,d)
        xs_patch = xs_patch.reshape(5*n_query,-1,n_shot,self.k,d).reshape(5*n_query,-1,n_shot*self.k,d)
        xq_patch = xq_patch.reshape(5*n_query,-1,n_shot,self.k,d).reshape(5*n_query,-1,n_shot*self.k,d)

        xs = xs.reshape(5*n_query,-1,n_shot,d).mean(2)     # [75,5,384]
        xq = xq.reshape(5*n_query,-1,n_shot,d).mean(2)     # [75,5,384]

        #-------------------------特征度量------------------------------   
        sf_patch = xs_patch.reshape(-1,self.shot*self.k,xs.shape[-1])
        qf_patch = xq_patch.reshape(-1,self.shot*self.k,xs.shape[-1])
        
        outputs = compute_logits(xq, xs, metric=self.metric, temp=self.temp)  # cos  10
        outputs_ = outputs.reshape(-1,outputs.shape[-1])
        outputs = outputs.sum(1)

        qr = xq.mean(1).unsqueeze(1)
        s_sim = compute_logits(qr,xs, metric=self.metric, temp=self.temp).squeeze(1)

        outputs_patch = compute_logits(qf_patch, sf_patch, metric=self.metric, temp=self.temp)
        outputs_patch = outputs_patch.mean(-1).mean(-1)
        
        outputs_patch = outputs_patch.reshape(outputs.shape[0],outputs.shape[1])
        outputs = self.weights[0]*outputs + self.weights[1]*outputs_patch + self.weights[2]*s_sim      #先累加计算outputs,在计算loss    86.78%

        return outputs,outputs_
    
    def forward_head(self, x, pre_logits: bool = False):
        return x if pre_logits else self.head(x)
    

def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model