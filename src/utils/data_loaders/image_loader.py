import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_files) 
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
