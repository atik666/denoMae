<p align="center">
  <img src="denoMAE.jpeg" alt="DenoMAE Logo" width="300"/>
</p>

# DenoMAE: A Multimodal Autoencoder for Denoising Modulation Signals

DenoMAE is a novel multimodal autoencoder framework that extends masked autoencoding for denoising and classifying modulation signals. It achieves state-of-the-art accuracy in automatic modulation classification with significantly reduced data requirements, while exhibiting robust performance across varying Signal-to-Noise Ratios (SNRs).

## Project Overview

DenoMAE consists of two main phases:
1. **Pretraining**: Training a masked autoencoder to reconstruct signals from partially masked inputs
2. **Fine-tuning**: Using the pretrained model for classification tasks on constellation signals

The model is based on Vision Transformer (ViT) architecture, adapted for signal processing applications.

## Directory Structure

```
DenoMAE_clean/
├── models/                  # Saved model weights
├── sample_generation/       # Scripts for generating training samples
├── datagen.py               # Data generator script
├── encoderDecoder.py        # Encoder-decoder implementation
├── DenoMAE.py               # DenoMAE backbone script
├── deno_Main.py             # Pretraining script
├── deno_finetune.py         # Fine-tuning script for classification
├── run_denomae.sh           # Shell script to run pretraining
├── run_finetune.sh          # Shell script to run fine-tuning
```

## Requirements

- Python 3.7++
- PyTorch 1.7+
- CUDA-capable GPU (recommended)
- Additional requirements: numpy, matplotlib, scikit-learn

```bash
pip install -r requirements.txt
```

## Usage

### Pretraining

To pretrain the DenoMAE model:

```bash
./run_pretrain.sh
```

This trains the model to reconstruct signals from masked inputs, creating robust feature representations.

### Fine-tuning

To fine-tune the pretrained model for classification:

```bash
./run_finetune.sh
```

The fine-tuning process adapts the pretrained model for specific classification tasks on constellation signals.

## Model Architecture

DenoMAE uses a Vision Transformer (ViT) architecture with:
- A patch embedding layer that converts input signals into tokens
- An encoder with self-attention layers to create latent representations
- A decoder that reconstructs the original signal from the encoded representation
- Classification head for fine-tuning tasks

## Sample Generation

The `sample_generation` folder contains scripts for generating synthetic training samples for modulation classification and denoising. The framework supports a variety of modulation schemes and noise levels, allowing for comprehensive model training and evaluation.

### Samples Generation Process

The sample generation follows these steps:

1. **Symbol Sequence Generation**: Random symbol sequences are generated according to the specified modulation type
2. **Modulation**: Symbols are modulated using the appropriate scheme (e.g., GMSK modulation for GMSK)
3. **Channel Effects**: Effects like phase offset are applied to simulate real-world impairments
4. **Noise Addition**: Additive White Gaussian Noise (AWGN) is added at configurable SNRs
5. **Visualization**: Constellation diagrams are rendered as images for processing by DenoMAE

### Constellation Image Generation

Constellation images are generated with a multi-scale approach:
- Each image uses RGB channels to represent different block sizes (5, 25, 50 pixels)
- Each point in the constellation is rendered as a Gaussian blob
- The intensity depends on the distance from the sample point
- This approach provides a rich visual representation of signal characteristics

### Unlabeled Sample Generation

For the pretraining phase, DenoMAE requires unlabeled samples which are generated using:

- **generateSamples.py**: Orchestrates the generation of unlabeled sample sets
- **funcSampleGeneration.py**: Contains core functions for creating constellation images and signals

Unlike the labeled samples used for fine-tuning, unlabeled samples include:

1. **Noiseless Images**: Clean constellation diagrams without noise
2. **Noisy Images**: Constellation diagrams with AWGN applied
3. **Raw Signal Data**: 
   - Noiseless signals (original modulated symbols)
   - Noise vectors (the added AWGN)
   - Noisy signals (noiseless signals with AWGN)

This paired data enables the model to learn the relationship between clean and corrupted signals during pretraining, which is crucial for the masked autoencoder's ability to denoise and reconstruct signals effectively.

To generate unlabeled samples:

```bash
cd sample_generation
python generateSamples.py
```

The unlabeled samples follow a similar organization structure but with different subdirectories:

```
unlabeled_small/
├── train/
│   ├── noiseLessImg/       # Clean constellation images
│   ├── noisyImg/           # Noisy constellation images
│   ├── noiselessSignal/    # Raw clean signals (numpy arrays)
│   ├── noise/              # Raw noise vectors (numpy arrays)
│   └── noisySignal/        # Raw noisy signals (numpy arrays)
└── test/                   # Same structure as train
```

These unlabeled samples are vital for the self-supervised pretraining of DenoMAE, where it learns to reconstruct masked portions of input signals.

### Usage

Sample generation can be customized with various parameters:

```python
CONFIG = {
    'samples_per_image': 1024,    # Number of symbols per constellation image
    'image_size': (224, 224),     # Output image dimensions
    'image_num': [100],           # Number of images per modulation type
    'mod_types': ['OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 
                  'GFSK', '4PAM', 'DQPSK', '16PAM', 'GMSK'],
    'mode': 'train',              # 'train' or 'test' 
    'SNR_dB': 0.0,                # Signal-to-noise ratio in dB
    'base_path': '...'            # Output directory path
}
```

To generate labeled samples, run:

```bash
cd sample_generation
python generateLabeledSamples.py
```

### Data Organization

Generated samples are organized in the following structure:

```
labeled/
├── {SNR_dB}_dB/            # Folders for different SNR levels
│   ├── train/              # Training samples
│   │   ├── OOK/            # Folders for each modulation type
│   │   ├── 4ASK/
│   │   └── ...
│   └── test/               # Testing samples
│       ├── OOK/
│       ├── 4ASK/
│       └── ...
└── files.txt               # Log of all generated files
```

This structured data generation approach enables comprehensive training and evaluation of DenoMAE across various modulation schemes and noise conditions, supporting both the pretraining and fine-tuning phases of the model.

