import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # type: ignore
import numpy as np
import os
from signal_loader import CustomSignalDataset
import random

def noise_generation(signal, snr_db):
    signal_power = np.mean(signal**2) # calculate signal power
    #calculate noise power
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power/snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return noise

class NoiseLoader(Dataset):
    def __init__(self, signal_dir, snr_db=None, transform=None):
        self.signal_dir = signal_dir
        self.transform = transform
        self.signal_files = os.listdir(signal_dir)
        self.snr_db = snr_db

    def __len__(self):
        return len(self.signal_files)
    
    def __getitem__(self, index):
        signal_path = os.path.join(self.signal_dir, self.signal_files[index])
        signal = np.load(signal_path)

        if self.transform:
            signal = self.transform(signal)

        if not self.snr_db:
            snr_db = random.uniform(-10, 10)
        
        noise = noise_generation(signal, snr_db)

        noisy_signal = signal + noise

        return noisy_signal, noise
    
class CustomNoiseDataset(Dataset):
    def __init__(self, noise_dir):
        self.noise_dir = noise_dir
        self.noise_files = os.listdir(noise_dir)

    def __len__(self):
        return len(self.noise_files) 
    
    def __getitem__(self, index):
        noise_path = os.path.join(self.noise_dir, self.noise_files[index])
        noise = np.load(noise_path)

        if len(noise) > 1024:
            noise = np.interp(np.linspace(0, len(noise) - 1, 1024), np.arange(len(noise)), noise)

        return noise