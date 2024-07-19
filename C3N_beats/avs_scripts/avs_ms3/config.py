from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.BATCH_SIZE = 4 # default 16
cfg.LAMBDA_1 = 5 # default: 5
cfg.MASK_NUM = 1 # 5 for fully supervised, 1 for weakly supervised

###############################
# TRAIN
cfg.TRAIN = edict()


# cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = False

cfg.TRAIN.PRETRAINED_PVTV2_PATH = "/data/zhaofeng/AVS_data/pretrained_backbones/pvt_v2_b5.pth"
cfg.TRAIN.PRETRAINED_PANNS_PATH = "/data/zhaofeng/AVS_data/pretrained_backbones/Cnn14_16k_mAP=0.438.pth"

cfg.TRAIN.PRETRAINED_BEATS_PATH = "/data/zhaofeng/AVS_data/pretrained_backbones/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"

cfg.TRAIN.PRETRAINED_SWIN_PATH = "/data/zhaofeng/AVS_data/pretrained_backbones/swin_base_patch4_window7_224.pth"

cfg.TRAIN.AV_SIM_MATRIX = "/data/zhaofeng/AVS_data/pretrained_backbones/similarity_matrix.h5"

cfg.TRAIN.FINE_TUNE_SSSS = False
cfg.TRAIN.PRETRAINED_S4_aAVS_WO_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"
cfg.TRAIN.PRETRAINED_S4_AVS_WITH_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = "/data/zhaofeng/AVS_data/avsbench_data/Multi-sources/ms3_meta_data.csv"
cfg.DATA.DIR_IMG = "/data/zhaofeng/AVS_data/avsbench_data/Multi-sources/ms3_data/visual_frames"
cfg.DATA.DIR_AUDIO_LOG_MEL = "/data/zhaofeng/AVS_data/avsbench_data/Multi-sources/ms3_data/audio_log_mel"
cfg.DATA.DIR_AUDIO = "/data/zhaofeng/AVS_data/avsbench_data/Multi-sources/ms3_data/audio_wav"
cfg.DATA.DIR_MASK = "/data/zhaofeng/AVS_data/avsbench_data/Multi-sources/ms3_data/gt_masks"
cfg.DATA.IMG_SIZE = (224, 224)

cfg.DATA.SR = 16000
###############################


if __name__ == "__main__":
    print(cfg)
    pdb.set_trace()