import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader1 import S4Dataset

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb

from PANNs.models import Cnn14_16k,Cnn14_emb128,Cnn14_emb512


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()

        self.audio_backbone = Cnn14_16k(sample_rate=cfg.DATA.SR, window_size=512,hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        self.pretrained = torch.load(cfg.TRAIN.PRETRAINED_PANNS_PATH)
        self.audio_backbone.load_state_dict(self.pretrained['model'])

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)['embedding'].data
        audio_cls = self.audio_backbone(audio)['clipwise_output'].data

        return audio_fea,audio_cls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--visual_backbone", default="pvt", type=str, help="use pvt-v2 or swin as the visual backbone")

    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3')

    parser.add_argument("--weights",type=str)
    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument('--log_dir', default='./test_logs', type=str)

    args = parser.parse_args()


    if (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    elif (args.visual_backbone).lower() == "swin":
        from model import Swin_AVSModel as AVSModel
        print('==> Use swin as the visual backbone...')
    else:
        raise NotImplementedError("only support the pvt-v2/swin")



    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir


    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages)
    
    weights = torch.load(args.weights,map_location='cpu')
    
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    
    model.load_state_dict(weights_dict)

    # model.load_state_dict(torch.load(args.weights))
    model = model.cuda()
    logger.info('=> Load trained model %s'%args.weights)

    # audio backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Test data
    split = 'test'
    test_dataset = S4Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Test
    model.eval()
    # with torch.no_grad():
    for n_iter, batch_data in enumerate(test_dataloader):
        imgs, audio, mask, category_list, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

        imgs = imgs.cuda()
        audio = audio.cuda()
        mask = mask.cuda()
        B, frame, C, H, W = imgs.shape
        imgs = imgs.view(B*frame, C, H, W)
        mask = mask.view(B*frame, H, W)
        audio = audio.view(-1, audio.shape[2])#, audio.shape[3], audio.shape[4])
        with torch.no_grad():
            audio_feature,audio_cls = audio_backbone(audio) # [B*T, 128]

        output = model(imgs, audio_feature, audio_cls) # [5, 1, 224, 224] = [bs=1 * T=5, 1, 224, 224]
        output = output.detach()
        
        if args.save_pred_mask:
            mask_save_path = os.path.join(log_dir, 'pred_masks')
            save_mask(output.squeeze(1), mask_save_path, category_list, video_name_list)

        miou = mask_iou(output.squeeze(1), mask)
        avg_meter_miou.add({'miou': miou})
        F_score = Eval_Fmeasure(output.squeeze(1), mask, log_dir)
        avg_meter_F.add({'F_score': F_score})
        print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))


    miou = (avg_meter_miou.pop('miou'))
    F_score = (avg_meter_F.pop('F_score'))
    print('test miou:', miou.item())
    print('test F_score:', F_score)
    logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))












