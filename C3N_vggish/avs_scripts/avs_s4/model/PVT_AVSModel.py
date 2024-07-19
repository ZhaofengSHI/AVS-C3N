import torch
import torch.nn as nn
from model.pvt import pvt_v2_b5
from model.TPAVI import TPAVIModule
import pdb
import h5py
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model.grad_cam_utils import ResizeTransform,ActivationsAndGradients,get_loss,compute_cam_per_layer
from model.utils import semantic_sim, cs_attention

    
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x

class Pred_endecoder(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, channel=256, config=None, vis_dim=[64, 128, 320, 512], tpavi_stages=[]):
        super(Pred_endecoder, self).__init__()
        self.cfg = config
        self.tpavi_stages = tpavi_stages
        self.vis_dim = vis_dim
        
        # visual-audio class similarity matrix
        self.sim_matrix = torch.tensor(h5py.File(self.cfg.TRAIN.AV_SIM_MATRIX, 'r')['similarity_matrix']).cuda()

        self.encoder_backbone = pvt_v2_b5()
        # self.target_layers = [self.encoder_backbone.norm1,self.encoder_backbone.norm2,self.encoder_backbone.norm3,self.encoder_backbone.norm4] # similar effect
        self.target_layers = [self.encoder_backbone.block1[-1],
                              self.encoder_backbone.block2[-1],
                              self.encoder_backbone.block3[-1],
                              self.encoder_backbone.block4[-1]]
        
        ### grad_cam init ###
        self.reshape_transform = ResizeTransform(im_h=224, im_w=224)
        self.activations_and_grads = ActivationsAndGradients(
            self.encoder_backbone, self.target_layers, self.reshape_transform)
        #####
        self.relu = nn.ReLU(inplace=True)
        # ASPP
        self.conv4 = ASPP(in_channel=self.vis_dim[3], depth=channel)
        self.conv3 = ASPP(in_channel=self.vis_dim[2], depth=channel)
        self.conv2 = ASPP(in_channel=self.vis_dim[1], depth=channel)
        self.conv1 = ASPP(in_channel=self.vis_dim[0], depth=channel)

        # decode path
        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)
        
        # tpavi
        self.tpavi_4 = TPAVIModule(in_channels=channel, mode='dot')
        self.tpavi_3 = TPAVIModule(in_channels=channel, mode='dot')
        self.tpavi_2 = TPAVIModule(in_channels=channel, mode='dot')
        self.tpavi_1 = TPAVIModule(in_channels=channel, mode='dot')

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        if self.training:
            self.initialize_pvt_weights()


    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        x = x.reshape(-1, 5, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x


    def forward(self, x, audio_feature, a_cls):
        
        # (x1, x2, x3, x4), v_cls = self.encoder_backbone(x)
        (x1, x2, x3, x4), v_cls = self.activations_and_grads(x)
        # visual-audio semantic similarity matrix and target category
        target_category = semantic_sim(v_cls,a_cls,self.sim_matrix)
        
        ### grad CAM ###
        assert (len(target_category) == x.size(0))
        
        self.encoder_backbone.zero_grad()
        loss = get_loss(v_cls, target_category)
        loss.backward(retain_graph=True)
        _, weight_per_layer = compute_cam_per_layer(self.activations_and_grads,x)
        
        weight1 = torch.tensor(weight_per_layer[0]).cuda()
        weight2 = torch.tensor(weight_per_layer[1]).cuda()
        weight3 = torch.tensor(weight_per_layer[2]).cuda()
        weight4 = torch.tensor(weight_per_layer[3]).cuda()
        
        # cam weight c s att
        x1 = cs_attention(x1,weight1)
        x2 = cs_attention(x2,weight2)
        x3 = cs_attention(x3,weight3)
        x4 = cs_attention(x4,weight4)
        
        #####################
        # shape for pvt-v2-b5
        # BF x  64 x 56 x 56
        # BF x 128 x 28 x 28
        # BF x 320 x 14 x 14
        # BF x 512 x  7 x  7    
        # ASPP
        conv1_feat = self.conv1(x1)  # BF x 256 x 56 x 56
        conv2_feat = self.conv2(x2)  # BF x 256 x 28 x 28
        conv3_feat = self.conv3(x3)  # BF x 256 x 14 x 14
        conv4_feat = self.conv4(x4)  # BF x 256 x  7 x  7
        
        # audio reshape
        audio = audio_feature.view(-1, 5, audio_feature.shape[-1])
        
        # 1
        conv1_feat = self.pre_reshape_for_tpavi(conv1_feat) 
        feature_map1, _ = self.tpavi_1(conv1_feat, audio) # [B, C, T, H, W]
        feature_map1 = self.post_reshape_for_tpavi(feature_map1)
        
        # 2
        conv2_feat = self.pre_reshape_for_tpavi(conv2_feat) 
        feature_map2, _ = self.tpavi_2(conv2_feat, audio) # [B, C, T, H, W]
        feature_map2 = self.post_reshape_for_tpavi(feature_map2)
        
        # 3
        conv3_feat = self.pre_reshape_for_tpavi(conv3_feat) 
        feature_map3, _ = self.tpavi_3(conv3_feat, audio) # [B, C, T, H, W]
        feature_map3 = self.post_reshape_for_tpavi(feature_map3)
        
        # 4
        conv4_feat = self.pre_reshape_for_tpavi(conv4_feat)
        feature_map4, _ = self.tpavi_4(conv4_feat, audio) # [B, C, T, H, W]
        feature_map4 = self.post_reshape_for_tpavi(feature_map4) 

        conv4 = self.path4(feature_map4)   # BF x 256 x 14 x 14
        conv43 = self.path3(conv4, feature_map3)    # BF x 256 x 28 x 28
        conv432 = self.path2(conv43, feature_map2)     # BF x 256 x 56 x 56
        conv4321 = self.path1(conv432, feature_map1)     # BF x 256 x 112 x 112

        pred = self.output_conv(conv4321)   # BF x 1 x 224 x 224

        return pred


    def initialize_pvt_weights(self,):

        pretrained_state_dicts = torch.load(self.cfg.TRAIN.PRETRAINED_PVTV2_PATH,map_location='cpu')
        self.encoder_backbone.load_state_dict(pretrained_state_dicts)
        
        print(f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {self.cfg.TRAIN.PRETRAINED_PVTV2_PATH}')
        
        # for name, param in self.encoder_backbone.named_parameters():
        #     # if "head" in name:
        #     param.requires_grad = True
                
        # pdb.set_trace()


if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    audio = torch.randn(2, 5, 128)
    # model = Pred_endecoder(channel=256)
    model = Pred_endecoder(channel=256, tpavi_stages=[0,1,2,3], tpavi_va_flag=True,)
    # output = model(imgs)
    output = model(imgs, audio)
    pdb.set_trace()