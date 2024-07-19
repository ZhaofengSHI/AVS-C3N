import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cs_attention(x, weights):

    # norm
    weights = F.normalize(weights,p=2,dim=1)
    
    channel_weights = torch.sigmoid(weights)
    x = x * channel_weights
    spatial_weights = torch.sigmoid(torch.mean(x, dim=1, keepdim=True))
    x = x * spatial_weights
    
    return x

def semantic_sim(v_cls,a_cls,sim_matrix,alpha = 0.1): 

    v_cls = v_cls.detach()
    a_cls = a_cls.detach()   
    B,_ = v_cls.shape
    
    # similarity matrix
    sim_matrix = sim_matrix.unsqueeze(0).repeat(B,1,1)
    
    # v_score matrix
    v_score = F.softmax(v_cls,dim=-1) ** alpha
    v_score_matrix = v_score.unsqueeze(2).repeat(1,1,sim_matrix.size(2))
    
    # a score matrix
    a_score = F.softmax(a_cls,dim=-1) ** alpha
    a_score_matrix = a_score.unsqueeze(1).repeat(1,sim_matrix.size(1),1)

    # W(i,j) = P_a(i) * S(i,j) * P_v(j)
    weight_matrix =  a_score_matrix * sim_matrix * v_score_matrix
    
    # generaye class list
    target_category = []
    for i in range(len(weight_matrix)):
        matrix = weight_matrix[i,:,:]
        max = torch.max(matrix)
        x,y = torch.where(matrix == max)
        activate_class = int(x[0])
        target_category.append(activate_class)
    
    return target_category
