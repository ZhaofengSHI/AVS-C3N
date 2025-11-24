# AVS-C3N
The codes of the paper "Cross-modal Cognitive Consensus guided Audio-Visual Segmentation"

# MS3

training (PVT)

```
cd avs_ms3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=11221 train.py --num_gpus=4 --session_name MS3_pvt_aug --visual_backbone pvt --max_epoches 60 --train_batch_size 1 --lr 0.0001 --tpavi_stages 0 1 2 3
```

testing (PVT)
```
cd avs_ms3
python test.py --session_name MS3_pvt --visual_backbone pvt --weights "path/to/weights" --test_batch_size 1 --tpavi_stages 0 1 2 3 --save_pred_mask
```

# S4

training (PVT)

```
cd avs_s4
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=11122 train.py --num_gpus=4 --session_name S4_pvt_aug --visual_backbone pvt --max_epoches 20 --train_batch_size 2 --lr 0.0001 --tpavi_stages 0 1 2 3
```

testing (PVT)
```
cd avs_s4
python test.py --session_name S4_pvt --visual_backbone pvt --weights "path/to/weights" --test_batch_size 1 --tpavi_stages 0 1 2 3 --save_pred_mask

```

The model is trained based on the AVSBench dataset https://github.com/OpenNLPLab/AVSBench


You can access the pre-processed data by contacting us by e-mail: zfshi@std.uestc.edu.cn
