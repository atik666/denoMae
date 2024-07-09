import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # type: ignore
import numpy as np
import os

class CustomSignalDataset(Dataset):
    def __init__(self, signal_dir, transform=None):
        self.signal_dir = signal_dir
        self.transform = transform
        self.signal_files = os.listdir(signal_dir)

    def __len__(self):
        return len(self.signal_files) 
    
    def __getitem__(self, index):
        signal_path = os.path.join(self.signal_dir, self.signal_files[index])
        signal = np.load(signal_path)

        if len(signal) > 1024:
            signal = np.interp(np.linspace(0, len(signal) - 1, 1024), np.arange(len(signal)), signal)

        if self.transform:
            signal = self.transform(signal)

        return signal