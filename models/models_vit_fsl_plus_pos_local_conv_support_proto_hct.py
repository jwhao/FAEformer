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

#========================== Define an image-to-class layer ==========================#


class ImgtoClass_Metric(nn.Module):
	def __init__(self, neighbor_k=3):
		super(ImgtoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k


	# Calculate the k-Nearest Neighbor of each local descriptor 
	def cal_cosinesimilarity(self, input1, input2):
		B, C, h, w = input1.size()
		Similarity_list = []

		for i in range(B):
			query_sam = input1[i]
			query_sam = query_sam.view(C, -1)
			query_sam = torch.transpose(query_sam, 0, 1)
			query_sam_norm = torch.norm(query_sam, 2, 1, True)   
			query_sam = query_sam/query_sam_norm

			if torch.cuda.is_available():
				inner_sim = torch.zeros(1, len(input2)).cuda()

			for j in range(len(input2)):
				support_set_sam = input2[j]
				support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
				support_set_sam = support_set_sam/support_set_sam_norm

				# cosine similarity between a query sample and a support category
				innerproduct_matrix = query_sam@support_set_sam

				# choose the top-k nearest neighbors
				topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)
				inner_sim[0, j] = torch.sum(topk_value)

			Similarity_list.append(inner_sim)

		Similarity_list = torch.cat(Similarity_list, 0)    

		return Similarity_list 


	def forward(self, x1, x2):

		Similarity_list = self.cal_cosinesimilarity(x1, x2)

		return Similarity_list

class ResidualDilatedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(384, 384, 3, stride=2, dilation=3, padding=3)
    
    def forward(self, x):
        return x[:,:,::2] + self.conv(x)

class SequenceCompressor(nn.Module):

    def __init__(self, original_seq_len, K):
        super().__init__()
        self.k = K
        self.linear = nn.Linear(original_seq_len, K)
    # 这个线性层将seq_len从original_seq_len映射到K，对每个特征独立处理
    def forward(self, x):
        # 先将x的shape转为 [batch*feat, seq]
        batch, feat, seq = x.shape
        x = x.reshape(batch * feat, seq)
        x = self.linear(x)  # [batch*feat, K]
        x = x.reshape(batch, feat, self.k)
        return x    

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, K):
        super().__init__()
        self.K = K
        self.query = nn.Parameter(torch.randn(K, feature_dim))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch, seq, feat]
        # query: [K, feat]
        # 计算注意力分数
        x = x.permute(0, 2, 1)  # [batch, feat, seq]-->[batch, seq, feat]
        attn_scores = torch.matmul(x, self.query.T)  # [batch, seq, K]
        attn_weights = self.softmax(attn_scores)  # [batch, seq, K]
        # 加权平均
        output = torch.matmul(attn_weights.transpose(1,2), x)  # [batch, K, feat]
        output = output.permute(0, 2, 1)  #[batch, K, feat] --> [batch, feat, K]

        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
#这种方法会为每个K生成一个查询向量，计算每个位置与该查询的相关性，
# 然后进行加权平均，得到K个特征向量。这样，每个K位置对应不同的注意力头，
# 聚合相关信息。这种方法可能更有效地减少不相关的干扰，因为它可以学习关注重要的部分。    

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,params=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.query = params.n_query
        self.shot = params.n_shot
        self.k = params.k        # ----------98 [224,224]
        self.metric = params.metric
        self.type = params.conv_type
        self.temp=params.temp
        self.w = params.weights   #---------default 2
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.pos = nn.Parameter(torch.zeros(1, 2*(self.k+1), kwargs['embed_dim']))
        self.weights = nn.Parameter(torch.ones(self.w))  # 初始权重为[1, 1, 1]
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
        elif self.type == 'pool':
            # 池化层：使用自适应平均池化
            self.conv1d = nn.AdaptiveAvgPool1d(self.k)
        elif self.type == 'linear':
            # 使用线性层进行降维
            self.conv1d = SequenceCompressor(196, self.k)
        elif self.type == 'mlp':
            # 使用多层线性层进行降维
            self.conv1d = Mlp(196, 9, self.k)   #self.k
        elif self.type == 'attn':
            # 使用线性层进行降维
            self.conv1d = AttentionPooling(384, self.k)

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
                xs = x[:,:n_shot].mean(1)    # [5,197,384]-----------------------------------------support proto
                
                xq = x[:,n_shot:].reshape(-1,p,d)    # [75,197,384]
                
                # xs_ = xs[:, 0, :] 
                # xq_ = xq[:, 0, :]
                #---------------------
                # 求解cls与每一个patch的相似度
                query_class = xq[:, 0, :].unsqueeze(1)  # Q x 1 x C
                query_image = xq[:, 1:, :]  # Q x L x C
                # query_image_ = query_image.unsqueeze(1).repeat(1,5,1,1).reshape(-1,p-1,xs.shape[-1])

                support_class = xs[:, 0, :].unsqueeze(1)  # KS x 1 x C
                support_image = xs[:, 1:, :]  # KS x L x C
                # support_image_ = support_image.unsqueeze(0).repeat(5*n_query,1,1,1).reshape(-1,p-1,xs.shape[-1])


                # ---------通过一维卷积压缩序列长度   196---> 98------------
                query_image = query_image.permute(0,2,1)
                support_image = support_image.permute(0,2,1)
                query_image_select = self.conv1d(query_image)
                support_image_select = self.conv1d(support_image)
                query_image_select = query_image_select.permute(0,2,1)
                support_image_select = support_image_select.permute(0,2,1)

                xs = torch.cat((support_class,support_image_select),dim=1)
                xq = torch.cat((query_class,query_image_select),dim=1)

                # xs_self = torch.cat([xs,xs],dim=1)
                # xq_self = torch.cat([xq,xq],dim=1)
                # x_self = torch.cat([xs_self,xq_self],dim=0)

                xs = xs.unsqueeze(0).repeat(5*n_query,1,1,1)
                xq = xq.unsqueeze(1).repeat(1,5,1,1)   #-------------------------------------------------------> 5 *n_shot--> 5
                
                x = torch.cat((xs,xq),dim=2).reshape(-1,(self.k+1)*2,d)

                # x = torch.cat([x_self,x],dim=0)

                x = x + self.pos   # ------------------加上新的位置编码
                
        # self.k = 98
        #-------------------------获得特征------------------------------
        Ns = 5
        Nq = 5*n_query + Ns

        x = self.norm(x)

        # xs_self = x[:Ns,0]
        # xq_self = x[Ns:Nq,0]
        # xs = x[Nq:,0]
        # xq = x[Nq:,self.k+1]
        # xs_patch = x[Nq:,1:self.k+1]
        # xq_patch = x[Nq:,self.k+2:]

        xs = x[:,0]
        xq = x[:,self.k+1]
        xs_patch = x[:,1:self.k+1]
        xq_patch = x[:,self.k+2:]
            
        xs = xs.reshape(5*n_query,-1,d)
        xq = xq.reshape(5*n_query,-1,d) 
        xs_patch = xs_patch.reshape(5*n_query,-1,self.k,d)
        xq_patch = xq_patch.reshape(5*n_query,-1,self.k,d)

        xq_patch = xq_patch.mean(dim=1,keepdim=True).repeat(1,Ns,1,1)        # patch 同样取平均-------------------------patch-proto
        # xs_patch = xs_patch.reshape(5*n_query,-1,n_shot,self.k,d).reshape(5*n_query,-1,n_shot*self.k,d)
        # xq_patch = xq_patch.reshape(5*n_query,-1,n_shot,self.k,d).reshape(5*n_query,-1,n_shot*self.k,d)

        # xs = xs.reshape(5*n_query,-1,n_shot,d).mean(2)     # [75,5,384]
        # xq = xq.reshape(5*n_query,-1,n_shot,d).mean(2)     # [75,5,384]

        #-------------------------特征度量------------------------------   
        # sf_patch = xs_patch.reshape(-1,self.shot*self.k,xs.shape[-1])
        # qf_patch = xq_patch.reshape(-1,self.shot*self.k,xs.shape[-1])
        sf_patch = xs_patch.reshape(-1,self.k,xs.shape[-1])
        qf_patch = xq_patch.reshape(-1,self.k,xs.shape[-1])
        
        outputs = compute_logits(xq, xs, metric=self.metric, temp=self.temp)  # cos  10
        outputs_ = outputs.reshape(-1,outputs.shape[-1])
        outputs = outputs.sum(1)

        qr = xq.mean(1).unsqueeze(1)
        s_sim = compute_logits(qr,xs, metric=self.metric, temp=self.temp).squeeze(1)

        outputs_patch = compute_logits(qf_patch, sf_patch, metric=self.metric, temp=self.temp)  # patch特征如何度量？当前是取得对角，考虑取queryproto；DN4的模式？
        # outputs_patch = outputs_patch.mean(-1).mean(-1)
        outputs_patch = outputs_patch.max(-1)[0].mean(-1)        # 取每一个patch的最大值，然后再取平均 patch_proto_dn4
        
        outputs_patch = outputs_patch.reshape(outputs.shape[0],outputs.shape[1])

       
        # logits = compute_logits(xq_self, xs_self, metric=self.metric, temp=self.temp)
        # logits = compute_logits(xq_, xs_, metric=self.metric, temp=self.temp)
        # logits_p = compute_logits(query_image_, support_image_, metric=self.metric, temp=self.temp).max(-1)[0].mean(-1)
        # logits_p = logits_p.reshape(outputs.shape[0],outputs.shape[1])
        # # xs_self = xs_self.unsqueeze(0).repeat(xq.shape[0],1,1)   # [5,384]--> [NQ,5,384]
        # # logits2 = compute_logits(xq, xs_self, metric=self.metric, temp=self.temp)
        # # logits2 = logits2.sum(1)
        # logits2 = compute_logits(xq.mean(1), xs_self, metric=self.metric, temp=self.temp)

        # xq_self = xq_self.unsqueeze(1).repeat(1,5,1)
        # logits3 = compute_logits(xq_self, xs, metric=self.metric, temp=self.temp)
        # # logits3 = logits3.sum(1)
        # logits3 = logits3.mean(1)

        # outputs = self.weights[0]*outputs + self.weights[1]*outputs_patch + self.weights[2]*s_sim      #ori #先累加计算outputs,在计算loss    86.78%
        # outputs = self.weights[0]*s_sim + self.weights[1]*outputs_patch + self.weights[2]*logits2 + self.weights[3]*logits3 + self.weights[4]*logits  
        # outputs = self.weights[0]*s_sim + self.weights[1]*outputs_patch + self.weights[2]*logits  
        # outputs = self.weights[0]*outputs + self.weights[1]*outputs_patch + self.weights[2]*s_sim + self.weights[3]*logits
        # outputs = self.weights[0]*s_sim + self.weights[1]*outputs_patch
        # outputs = self.weights[0]*s_sim + self.weights[1]*outputs_patch + self.weights[2]*logits
        # outputs = self.weights[0]*s_sim + self.weights[1]*outputs_patch + self.weights[2]*logits + self.weights[3]*logits_p
        # outputs = self.weights[0]*s_sim + self.weights[1]*logits         # nopatch
        
        if self.w == 1:
             outputs = self.weights[0]*s_sim
        elif self.w == 2:
             outputs = self.weights[0]*s_sim + self.weights[1]*outputs_patch
        elif self.w == 3:
             outputs = self.weights[0]*s_sim + self.weights[1]*outputs_patch + self.weights[2]*logits
        elif self.w == 4:
             outputs = self.weights[0]*s_sim + self.weights[1]*outputs_patch + self.weights[2]*logits + self.weights[3]*logits_p
             
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
        patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
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