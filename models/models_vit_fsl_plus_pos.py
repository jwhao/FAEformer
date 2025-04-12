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

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,params=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.query = params.n_query
        self.shot = params.n_shot
        self.k = params.k
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.pos = nn.Parameter(torch.zeros(1, 2*(self.k+1), kwargs['embed_dim']))

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

                query_sim_between_cls_patch = compute_logits(query_class,query_image,metric = 'cos').squeeze()
                support_sim_between_cls_patch = compute_logits(support_class,support_image,metric = 'cos').squeeze()

                # 找出相似度大于某一阈值的索引及patch
                # 获取每一行大于阈值的布尔掩码----------是按照阈值掩码，还是选取前K个？？？
                query_image_select = patch_selection(query_sim_between_cls_patch,query_image,self.k)   # 挑选相似度排名前20的
                support_image_select = patch_selection(support_sim_between_cls_patch,support_image,self.k)   # 挑选相似度排名前20的

                xs = torch.cat((support_class,support_image_select),dim=1)
                xq = torch.cat((query_class,query_image_select),dim=1)
                xs = xs.unsqueeze(0).repeat(5*n_query,1,1,1)
                xq = xq.unsqueeze(1).repeat(1,5*n_shot,1,1)
                
                x = torch.cat((xs,xq),dim=2).reshape(-1,(self.k+1)*2,d)
                x = x + self.pos   # ------------------加上新的位置编码
                
        N = int(x.shape[0]/2) 
        if self.global_pool:
            xq = x[:N, 1:, :].mean(dim=1)  # global pool without cls token
            xs = x[N:, 1:, :].mean(dim=1)
            outcome_q = self.fc_norm(xq)
            outcome_s = self.fc_norm(xs)
        else:
            x = self.norm(x)
            xs = x[:,0]
            xq = x[:,self.k+1]
             
            xs = xs.reshape(5*n_query,-1,d)
            xq = xq.reshape(5*n_query,-1,d) 

            xs = xs.reshape(5*n_query,-1,n_shot,d).mean(2)     # [75,5,384]
            xq = xq.reshape(5*n_query,-1,n_shot,d).mean(2)     # [75,5,384]

        return xs, xq
    
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