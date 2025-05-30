import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F
from pathlib import Path

class DenoMAEDataGenerator(Dataset):
    def __init__(self, num_of_modalities, noisy_image_path, noiseless_img_path, noisy_signal_path, noiseless_signal_path, noise_path, 
                 image_size=(224, 224), target_length=50176, transform=None):
        """
        Initializes the data generator for DenoMAE.
        
        Args:
            num_of_modalities (int): Number of modalities to return.
            image_path (str): Directory containing image files (e.g., PNG).
            noiseLess_image_path (str): Directory containing noiseless image files (e.g., PNG).
            signal_path (str): Directory containing signal files.
            noise_path (str): Directory containing noise files.
            image_size (tuple): Size to which images will be resized.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.num_of_modalities = num_of_modalities
        self.image_size = image_size
        self.transform = transform or transforms.ToTensor()
        self.target_length = target_length

        # Use pathlib for more Pythonic path handling
        self.paths = {
            'noisy_image': Path(noisy_image_path),
            'noiseless_image': Path(noiseless_img_path),
            'noisy_signal': Path(noisy_signal_path),
            'noiseless_signal': Path(noiseless_signal_path),
            'noise': Path(noise_path)
        }
        
        # Use list comprehension for file listing
        self.filenames = {
            key: sorted(path.glob('*')) for key, path in self.paths.items()
        }

    def __len__(self):
        # Return the length of the largest list of filenames
        return max(len(filenames) for filenames in self.filenames.values())
    
    @staticmethod
    def preprocess_npy(npy_path, target_length, image_size):
        """
        Loads and preprocesses an NPY file.
        
        Args:
            npy_path (Path): Path to the NPY file.
            target_length (int): The length to which the data should be resized.
            image_size (tuple): The size to which the data should be resized.
        
        Returns:
            torch.Tensor: The processed NPY data as a PyTorch tensor.
        """
        npy_data = np.load(npy_path)
        if len(npy_data) != target_length:
            x = np.linspace(0, len(npy_data) - 1, target_length)
            npy_data = np.interp(x, np.arange(len(npy_data)), npy_data)

        npy_data = npy_data.reshape(1, image_size[0], image_size[1]).repeat(3, axis=0)
        npy_tensor = torch.from_numpy(npy_data.real).float().unsqueeze(0)
        return F.interpolate(npy_tensor, size=image_size, mode='bilinear', align_corners=False).squeeze(0)

    def __getitem__(self, index):
        # Load and preprocess image
        noisy_img = Image.open(self.filenames['noisy_image'][index]).resize(self.image_size)
        noisy_img = self.transform(noisy_img)

        noiseless_img = Image.open(self.filenames['noiseless_image'][index]).resize(self.image_size)
        noiseless_img = self.transform(noiseless_img)

        # Load and preprocess npy files
        noisy_signal = self.preprocess_npy(self.filenames['noisy_signal'][index], target_length=self.target_length, image_size=self.image_size)
        noiseless_signal = self.preprocess_npy(self.filenames['noiseless_signal'][index], target_length=self.target_length, image_size=self.image_size)
        noise_data = self.preprocess_npy(self.filenames['noise'][index], target_length=self.target_length, image_size=self.image_size)

        return_items = [noisy_img, noisy_signal, noiseless_img, noiseless_signal, noise_data], \
                [noiseless_img, noiseless_signal, noiseless_img, noiseless_signal, noise_data]
        
        return return_items[0][:self.num_of_modalities], return_items[1][:self.num_of_modalities]

if __name__ == "__main__":
    DenoMAEDataGenerator