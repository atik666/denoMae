#!/bin/bash

# Define variables for arguments
PRETRAINED_MODEL_PATH="models/MAE_0.105.pth"
DATA_PATH="/mnt/d/Constellation/labeled/9_dB"
SAVE_MODEL_PATH="models/classify_MAE.pth"
IMG_SIZE=224
PATCH_SIZE=16
IN_CHANS=3
EMBED_DIM=768
ENCODER_DEPTH=12
DECODER_DEPTH=4
NUM_HEADS=12
BATCH_SIZE=32
NUM_EPOCHS=150
LEARNING_RATE=1e-4
NUM_CLASSES=15
NUM_MODALITY=5
DEVICE="cuda:1"

# Run the Python script with arguments
python deno_finetune.py \
  --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --save_model_path "$SAVE_MODEL_PATH" \
  --img_size $IMG_SIZE \
  --patch_size $PATCH_SIZE \
  --in_chans $IN_CHANS \
  --embed_dim $EMBED_DIM \
  --encoder_depth $ENCODER_DEPTH \
  --decoder_depth $DECODER_DEPTH \
  --num_heads $NUM_HEADS \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --num_classes $NUM_CLASSES \
  --num_modality $NUM_MODALITY \
  --device "$DEVICE"
