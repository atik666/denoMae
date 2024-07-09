import os
import random
import pandas as pd
import numpy as np

from image_loader import CustomImageDataset
from signal_loader import CustomSignalDataset
from noise_loader import CustomNoiseDataset

from torch.utils.data import DataLoader
import torchvision.transforms as T
from itertools import islice
from PIL import Image

import torch

image_path = './data/noisyImg/'
signal_path = './data/signal/'
noise_path = './data/noise/'
batch_size = 1
image_size = 32

config = {
    'image_path':image_path,
    'signal_path':signal_path,
    'noise_path':noise_path,
    'batch_size': batch_size,
    'image_size': image_size,
}

# img_list = os.walk(config['image_path'])
# print(img_list)
# noise_list = os.walk(config['noise_path'])
# signal_path = os.walk(config['signal_path'])

# transform = T.Compose(T.ToTensor())

image = CustomImageDataset(config['image_path'])
noise = CustomSignalDataset(config['noise_path'])
signal = CustomSignalDataset(config['signal_path'])


data = {'image':image, 'signal':signal, 'noise':noise}

image_dataloader = DataLoader(data['image'], batch_size=config['batch_size'], shuffle=True)
print("image loader: ", image_dataloader)

def preprocess(image, signal, noise):

    # resized_image = image.resize(config['image_size'])
    # image_array = np.array(resized_image)
    normalized_image_array = image / 255.0

    # def norm_array(arr):
    #     return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    # def norm_tensor(signal):
    #     return (signal - torch.min(signal)) / (torch.max(signal) - torch.min(signal))
    
    def norm_complex_tensor(tensor):
        # Separate real and imaginary parts
        real_part = tensor.real
        imag_part = tensor.imag

        # Normalize real part
        real_min = torch.min(real_part)
        real_max = torch.max(real_part)
        real_norm = (real_part - real_min) / (real_max - real_min)

        # Normalize imaginary part
        imag_min = torch.min(imag_part)
        imag_max = torch.max(imag_part)
        imag_norm = (imag_part - imag_min) / (imag_max - imag_min)

        # Reconstruct the complex tensor
        normalized_tensor = torch.complex(real_norm, imag_norm)
        
        return normalized_tensor

    return normalized_image_array, norm_complex_tensor(signal), norm_complex_tensor(noise)

def generate_random_image_masks(images, mask_ratio=0.75):
    """
    Generate random masks for a batch of input images.

    Args:
        images (torch.Tensor): Batch of images with shape [batch_size, height, width, channels].
        mask_ratio (float): Ratio of the image pixels to be masked.

    Returns:
        torch.Tensor: Random masks with the same shape as the input images, including the channel dimension.
    """

    batch_size, H, W, C = images.size()  # Get dimensions from the input batch
    num_pixels = H * W
    num_masked = int(num_pixels * mask_ratio)

    masks = torch.ones((batch_size, H, W, C), dtype=torch.bool)

    for i in range(batch_size):
        for c in range(C):
            masked_indices = random.sample(range(num_pixels), num_masked)
            for idx in masked_indices:
                y = idx // W
                x = idx % W
                masks[i, y, x, c] = False
    
    return masks

def apply_masks(image, masks):
    """
    Apply masks to the input data.

    Args:
        input_data (torch.Tensor): The input data to be masked.
        masks (torch.Tensor): The masks to apply.

    Returns:
        torch.Tensor: The masked input data.
    """

    masked_data = image * (1 - masks.float())
    
    return masked_data

class MultiModalDataGenerator:
    def __init__(self,data, config, ):
        self.data = data
        self.config = config
        # generator process
        # load the signal
        # generate the noise
        # merge noise and signal
        # generate images of noise and signal and noisy signal
        # return (noisy_image, noise, clean_image)

        # self.normalized_image_array = self.preprocess()

        self.image_dataloader = DataLoader(self.data['image'], batch_size=config['batch_size'], shuffle=True)
        self.signal_dataloader = DataLoader(self.data['signal'], batch_size=config['batch_size'], shuffle=True)
        self.noise_dataloader = DataLoader(self.data['noise'], batch_size=config['batch_size'], shuffle=True)

    def __getitem__(self, index):

        image_batch = next(iter(self.image_dataloader))
        signal_batch = next(iter(self.signal_dataloader))
        noise_batch = next(iter(self.noise_dataloader))
    
        # image_batch = next(islice(self.image_dataloader, index, index+1))
        # signal_batch = next(islice(self.signal_dataloader, index, index+1))
        # noise_batch = next(islice(self.noise_dataloader, index, index+1))

        image, signal, noise = preprocess(image_batch, signal_batch, noise_batch)
        masks = generate_random_image_masks(image, mask_ratio=0.75)
        print("masks: ", masks.shape)
        masked_image = apply_masks(image, masks)

        print("masked_image: ", masks.shape)
        
        return masked_image


masked_image = MultiModalDataGenerator(data,config)[1]

tensor_np = masked_image.squeeze(0).numpy()
tensor_np = (tensor_np * 255).astype(np.uint8)
image = Image.fromarray(tensor_np)
image = image.resize((224, 224), Image.BILINEAR)
image.save("masked_image.jpg", dpi=(600,600))