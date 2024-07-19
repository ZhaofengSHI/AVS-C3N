import torch
from torchvggish import vggish
from torchvggish.vggish import Postprocessor
from torchvggish import vggish_input,vggish_params
import torch.nn as nn
import torch.nn.functional as F
import math

def uint8_to_float32(x):
    return (x.float() - 128.) / 128.

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.cfg = cfg
        self.audio_backbone = vggish.VGGish(self.cfg, device)
        
        freq_bins = 128
        classes_num = 527

        # Hyper parameters
        hidden_units = 1024
        drop_rate = 0.5
        
        # post_process
        print(self.cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH)
        self.pproc = Postprocessor()
        state_dict = torch.load(self.cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH,map_location='cpu')
                # TODO: Convert the state_dict to torch
        state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
            state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float
        )
        state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(
            state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
        )
        self.pproc.load_state_dict(state_dict,strict=True)
        
        # classification
        self.classification_model = FeatureLevelSingleAttention(
            freq_bins, classes_num, hidden_units, drop_rate).eval()
        
        self.initialize_head_weights()

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)

        # post
        audio_fea_post = self._postprocess(audio_fea)
        audio_fea_post = uint8_to_float32(audio_fea_post)
        audio_fea_post = audio_fea_post.unsqueeze(1)

        # classification
        audio_cls = self.classification_model(audio_fea_post)   
        
        return audio_fea,audio_cls
    
    def _postprocess(self, x):
        return self.pproc(x)
    
    def initialize_head_weights(self,):

        pretrained_state_dicts = torch.load(self.cfg.TRAIN.PRETRAINED_HEAD_PATH,map_location='cpu')
        self.classification_model.load_state_dict(pretrained_state_dicts)




def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.weight.data.fill_(1.)


class Attention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att,)
        init_layer(self.cla)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return F.sigmoid(x)

        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        return x
    

class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, hidden_units, drop_rate):
        super(EmbeddingLayers, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(
            in_channels=freq_bins, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv3 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.bn0 = nn.BatchNorm2d(freq_bins)
        self.bn1 = nn.BatchNorm2d(hidden_units)
        self.bn2 = nn.BatchNorm2d(hidden_units)
        self.bn3 = nn.BatchNorm2d(hidden_units)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)

        init_bn(self.bn0)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input, return_layers=False):
        """input: (samples_num, time_steps, freq_bins)
        """

        drop_rate = self.drop_rate

        # (samples_num, freq_bins, time_steps)
        x = input.transpose(1, 2)

        # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)
        x = x[:, :, :, None].contiguous()

        a0 = self.bn0(x)
        a1 = F.dropout(F.relu(self.bn1(self.conv1(a0))),
                       p=drop_rate,
                       training=self.training)

        a2 = F.dropout(F.relu(self.bn2(self.conv2(a1))),
                       p=drop_rate,
                       training=self.training)

        emb = F.dropout(F.relu(self.bn3(self.conv3(a2))),
                        p=drop_rate,
                        training=self.training)

        if return_layers is False:
            # (samples_num, hidden_units, time_steps, 1)
            return emb

        else:
            return [a0, a1, a2, emb]
        
class FeatureLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(FeatureLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        
        self.attention = Attention(
            hidden_units,
            hidden_units,
            att_activation='sigmoid',
            cla_activation='linear')
            
        self.fc_final = nn.Linear(hidden_units, classes_num)
        self.bn_attention = nn.BatchNorm1d(hidden_units)

        self.drop_rate = drop_rate

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)
        init_bn(self.bn_attention)

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps)
        """
        drop_rate = self.drop_rate

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, hidden_units)
        b2 = self.attention(b1)
        b2 = F.dropout(
            F.relu(
                self.bn_attention(b2)),
            p=drop_rate,
            training=self.training)
        
        # (samples_num, classes_num)
        # output = F.sigmoid(self.fc_final(b2))
        output = self.fc_final(b2)

        return output