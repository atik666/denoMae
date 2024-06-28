import os
import random
import pandas as pd
from image_loader import CustomImageDataset
from signal_loader import CustomSignalDataset
from noise_loader import CustomNoiseDataset

image_path = './data/noisyImg/'
signal_path = './data/signal/'
noise_path = './data/noise/'

config = {
    'image_path':image_path,
    'signal_path':signal_path,
    'noise_path':noise_path,
}

img_list = os.walk(config['image_path'])
print(img_list)
noise_list = os.walk(config['noise_path'])
signal_path = os.walk(config['signal_path'])

image = CustomImageDataset(config['image_path'])[0]
print(image)

signal = CustomSignalDataset(config['signal_path'])[0]
print(signal.shape)

noise = CustomNoiseDataset(config['noise_path'])[0]
print(noise.shape)
