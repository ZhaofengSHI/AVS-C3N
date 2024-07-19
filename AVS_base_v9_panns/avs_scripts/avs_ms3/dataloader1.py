import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms

from config import cfg
import pdb
import librosa
import random



def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        if img_tensor.size(0) == 3:
            trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = trans(img_tensor)
        return img_tensor
    return img_PIL




def load_audio(audio_wav_path,cut_time = 5.0):

    # Load audio
    (waveform, _) = librosa.core.load(audio_wav_path, sr=cfg.DATA.SR, mono=False)
    
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=0)

    max_length = int(cut_time * cfg.DATA.SR) #80000
    wav = np.zeros((max_length,))
    length = min(max_length,waveform.shape[0])
    
    wav[:length] = waveform[:length]

    wav = wav[None, :]    # (1, audio_length)
    wav = torch.Tensor(wav)
    wav = wav.view(int(cut_time), -1)

    return wav


class MS3Dataset(Dataset):
    """Dataset for multiple sound source segmentation"""
    def __init__(self, split='train'):
        super(MS3Dataset, self).__init__()
        self.split = split
        self.mask_num = 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))



    def __getitem__(self, index):
        
        p1 = random.randint(0,1)

        # seed = torch.random.seed()

        img_transform_train = transforms.Compose([
            transforms.Resize(cfg.DATA.IMG_SIZE),
            # transforms.RandomResizedCrop(cfg.DATA.IMG_SIZE, (0.5, 2.0)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomHorizontalFlip(p1),
            transforms.ToTensor(),
        ])   

        mask_transform_train = transforms.Compose([
            transforms.Resize(cfg.DATA.IMG_SIZE),
            # transforms.RandomResizedCrop(cfg.DATA.IMG_SIZE, (0.5, 2.0)),
            transforms.RandomHorizontalFlip(p1),
            transforms.ToTensor(),
        ])   


        img_transform_val = transforms.Compose([
            transforms.Resize(cfg.DATA.IMG_SIZE),
            transforms.ToTensor(),
        ])
    
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, video_name)
        # audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, video_name + '.pkl')
        audio_wav_path = os.path.join(cfg.DATA.DIR_AUDIO, self.split, video_name + '.wav')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, video_name)

        audio_waveform = load_audio(audio_wav_path)
        
        if self.split == 'train':
            imgs, masks = [], []

            for img_id in range(1, 6):

                # torch.random.manual_seed(seed)

                img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=img_transform_train)
                imgs.append(img)
            for mask_id in range(1, self.mask_num + 1):

                # torch.random.manual_seed(seed)

                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=mask_transform_train, mode='P')
                masks.append(mask)

            imgs_tensor = torch.stack(imgs, dim=0)
            masks_tensor = torch.stack(masks, dim=0)

        if self.split != 'train':

            imgs, masks = [], []
            for img_id in range(1, 6):
                img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=img_transform_val)
                imgs.append(img)
            for mask_id in range(1, self.mask_num + 1):
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=img_transform_val, mode='P')
                masks.append(mask)
            imgs_tensor = torch.stack(imgs, dim=0)
            masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_waveform, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)






# if __name__ == "__main__":
#     train_dataset = MSSSDataset('train')
#     train_dataloader = torch.utils.data.DataLoader(train_dataset,
#                                                      batch_size=2,
#                                                      shuffle=False,
#                                                      num_workers=8,
#                                                      pin_memory=True)

#     for n_iter, batch_data in enumerate(train_dataloader):
#         imgs, audio, mask, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
#         # imgs, audio, mask, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
#         pdb.set_trace()
#     print('n_iter', n_iter)
#     pdb.set_trace()