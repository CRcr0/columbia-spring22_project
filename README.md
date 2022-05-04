# columbia-spring22_project

Used pretrained up-detr_60_epoches model which was trained on ImageNet

Used mini coco 2017 and coco mini datasets to do the fine tuning.

to fine tune,

!python -m torch.distributed.launch --nproc_per_node=6 --use_env detr_main.py \
    --lr_drop 200 \
    --epochs 150 \
    --lr_backbone 5e-5 \
    --pre_norm \
    --coco_path path/to/coco \
    --pretrain path/to/save_model/up-detr-pre-training-60ep-imagenet.pth
    
   By running on 6 x RTX 3080, it costs 24 hour.
   
   The retrained model and log will be stored in the ~/checkpoint/ file.
   
   To evaluate the result,
   
   !python detr_main.py \
    --batch_size 1 \
    --eval \
    --no_aux_loss \
    --pre_norm \
    --coco_path path/to/coco \
    --resume checkpoint/checkpoint.pth
