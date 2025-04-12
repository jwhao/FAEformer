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


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        i = 0
        # n_query = 4 if int(x.shape[0]/5) < 10 else 10
        n_query = 1
        n_shot = int(x.shape[0]/5)-n_query
        for blk in self.blocks:
            x = blk(x)                                         # [30 =n_way*(n_shot+n_query),197,768]
            i += 1
            if i == 11:
                # n_shot = 5
                # n_query = 4 
                _,p,d = x.shape
                x = x.reshape(5,-1,p,d)                        # [5,6,197,768]
                xs = x[:,:n_shot].mean(1)                     # proto
                xs_cls = xs[:,0].unsqueeze(1)                                 # support only keep patch_token
                xs_patch = xs[:,1:]
                # xs = xs.unsqueeze(0).repeat(5* n_query,1,1,1)
                xq = x[:,n_shot:].reshape(-1,p,d)
                xq_cls = xq[:,0].unsqueeze(1)                                    # query only keep cls_token
                xq_patch = xq[:,1:]

                xs_cls = xs_cls.unsqueeze(0).repeat(5*n_query,1,1,1)    # [5,1,768] -> [1,5,1,768]-> [10,5,196,768]
                xq_patch = xq_patch.unsqueeze(1).repeat(1,5,1,1)        # [10,196,768] -> [10,1,196,768]-> [10,5,196,768]
                xs_cls_xq_patch = torch.cat((xs_cls,xq_patch),dim=2).reshape(-1,p,d)      # [10,5,197,768] -> [50,197,768]

                xq_cls = xq_cls.unsqueeze(1).repeat(1,5,1,1)            # [10,1,768] -> [10,1,1,768]-> [10,5,1,768]
                xs_patch = xs_patch.unsqueeze(0).repeat(5*n_query,1,1,1)   # [5,196,768] -> [1,5,196,768]-> [10,5,196,768]
                xq_cls_xs_patch = torch.cat((xq_cls,xs_patch),dim=2).reshape(-1,p,d)       # [10,5,197,768] -> [50,197,768]
                
                x = torch.cat((xs_cls_xq_patch,xq_cls_xs_patch),dim=0)
        N = int(x.shape[0]/2)
        if self.global_pool:
            xq = x[:N, 1:, :].mean(dim=1)  # global pool without cls token
            xs = x[N:, 1:, :].mean(dim=1)
            outcome_q = self.fc_norm(xq)
            outcome_s = self.fc_norm(xs)
        else:
            x = self.norm(x)
            
            outcome_q = x[N:, 0]
            
            outcome_s = x[:N, 0]

        return outcome_q, outcome_s
    
    def forward_head(self, x, pre_logits: bool = False):
        return x if pre_logits else self.head(x)


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