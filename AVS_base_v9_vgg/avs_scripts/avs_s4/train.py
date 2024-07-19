import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import random
import numpy as np
import torch
import argparse
import logging

from config import cfg
from dataloader1 import S4Dataset
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import pdb
from model.audio_model import audio_extractor
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from poly_lr import PolynomialLRDecay

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    parser.add_argument('--num_gpus', type=int, default=8, help='node rank for distributed training')
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--visual_backbone", default="pvt", type=str, help="use pvt-v2 or swin as the visual backbone")

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=1, type=int)

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3')

    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)

    args = parser.parse_args()

    # initialize devices
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    device = torch.device(f'cuda:{args.local_rank}')

    if (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    elif (args.visual_backbone).lower() == "swin":
        from model import Swin_AVSModel as AVSModel
        print('==> Use swin as the visual backbone...')
    else:
        raise NotImplementedError("only support the pvt-v2/swin")

    # Fix seed
    FixSeed = 1234
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

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
    base_model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages)
    
    model = base_model
    model = model.cuda()
    # convert sync bn
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # convert model to distributed model
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    
    model.train()

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Data
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=DistributedSampler(train_dataset,shuffle=True,seed=1234),
                                                        batch_size=args.train_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    max_step = (len(train_dataset) // (args.train_batch_size * args.num_gpus)) * args.max_epoches

    val_dataset = S4Dataset('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, args.lr) ##
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_dice_loss = pyutils.AverageMeter('dice_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')

    # poly lr scheduler
    scheduler = PolynomialLRDecay(optimizer,
                                 max_decay_steps=args.max_epoches - 5,
                                 end_learning_rate=1e-5,
                                 power=1.0)

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0

    for epoch in range(args.max_epoches):

        train_dataloader.sampler.set_epoch(epoch)

        for n_iter, batch_data in enumerate(train_dataloader):

            imgs, audio, mask, videoname = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()

            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B, H, W)

            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) # [B*T, 1, 96, 64]

            with torch.no_grad():
                audio_feature,audio_cls = audio_backbone(audio) # [B*T, 128]
                
            output = model(imgs, audio_feature, audio_cls) # [bs*5, 1, 224, 224]
            
            loss, loss_dict = IouSemanticAwareLoss(output, mask.unsqueeze(1))

            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
            avg_meter_dice_loss.add({'dice_loss': loss_dict['dice_loss']})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if torch.distributed.get_rank() == 0:  ###
                if (global_step-1) % 50 == 0:
                    train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, dice_loss:%.4f, lr: %.7f'%(
                                global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_dice_loss.pop('dice_loss'), optimizer.param_groups[0]['lr'])
                    logger.info(train_log)

        # scheduler
        if epoch < args.max_epoches-5:
            scheduler.step()  

        torch.cuda.empty_cache()
        # Validation:
        base_model.eval()
        # with torch.no_grad():
        for n_iter, batch_data in enumerate(val_dataloader):
            imgs, audio, mask, _, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B*frame, H, W)
            
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) # [B*T, 1, 96, 64]
            
            with torch.no_grad():
                audio_feature,audio_cls = audio_backbone(audio)

            output = base_model(imgs, audio_feature, audio_cls) # [bs*5, 1, 224, 224]
            output = output.detach()

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})

        miou = (avg_meter_miou.pop('miou'))
        if miou > max_miou:
            model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
            torch.save(base_model.state_dict(), model_save_path)
            best_epoch = epoch
            if torch.distributed.get_rank() == 0:  ###
                logger.info('save best model to %s'%model_save_path)    

        miou_list.append(miou)
        max_miou = max(miou_list)

        if torch.distributed.get_rank() == 0:  ###
            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
            logger.info(val_log)

        model.train()
        
    if torch.distributed.get_rank() == 0:  ###
        logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))