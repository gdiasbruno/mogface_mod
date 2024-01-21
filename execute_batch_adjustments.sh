#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_malf_finetune_val_epochs.py \
--batch_size 1 \
--config ./configs/mogface/hypothesis_null_configs/MogFace_malf_test.yml \
--num_workers 10 \
--pretrain_weights_full_model ./pretrain_weights/mogface_config_weight/model_140000.pth \
--unfreeze_layers pred_net \
--validation_epoch_gap 2 \
--logs_folder ./20_01_2023_null_hypothesis_test_fold_test


CUDA_VISIBLE_DEVICES=0 python train_malf_finetune_val_epochs.py \
--batch_size 1 \
--config ./configs/mogface/hypothesis_null_configs/MogFace_malf_test.yml \
--num_workers 10 \
--pretrain_weights_full_model ./pretrain_weights/mogface_config_weight/model_140000.pth \
--unfreeze_layers pred_net \
--validation_epoch_gap 2 \
--logs_folder ./20_01_2023_null_hypothesis_test_fold_test_2

runpodctl stop pod ktnzkb9ah9f9ka