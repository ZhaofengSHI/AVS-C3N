# AVS-C3N
The codes of the paper "Cross-modal Cognitive Consensus guided Audio-Visual Segmentation"

# MS3

training (PVT)

```
cd avs_ms3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=11221 train.py --num_gpus=4 --session_name MS3_pvt_aug --visual_backbone pvt --max_epoches 60 --train_batch_size 1 --lr 0.0001 --tpavi_stages 0 1 2 3
```

testing
```

```
