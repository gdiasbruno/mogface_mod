#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train_malf_finetune_val_epochs.py \
--batch_size 25 \
--config ./configs/mogface/hypothesis_null_configs/MogFace_malf_fold_01.yml \
--num_workers 10 \
--pretrain_weights_full_model ./pretrain_weights/mogface_config_weight/model_140000.pth \
--unfreeze_layers pred_net \
--validation_iter_gap 500 \
--logs_folder ./21_01_2023_null_hypothesis_test_fold_01
