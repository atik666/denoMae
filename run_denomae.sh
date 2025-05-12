#!/bin/bash

python3 deno_Main.py \
  --train_path "/mnt/d/Constellation/unlabeled_small/train/" \
  --test_path "/mnt/d/Constellation/unlabeled_small/test/" \
  --batch_size 32 \
  --image_size 224 224 \
  --patch_size 16 \
  --in_chans 3 \
  --mask_ratio 0.75 \
  --embed_dim 768 \
  --encoder_depth 12 \
  --decoder_depth 4 \
  --num_heads 12 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --log_dir "runs/multimae" \
  --model_dir "models" \
  --num_modality 5 \
  --model_name "MAE" \
  --final_model_name "MAE_mask" \
  --n_workers 4 \
  "$@"