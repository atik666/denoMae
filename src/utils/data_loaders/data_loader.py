import os
import random
import pandas as pd
import numpy as np

from image_loader import CustomImageDataset
from signal_loader import CustomSignalDataset
from noise_loader import CustomNoiseDataset

image_path = './data/noisyImg/'
signal_path = './data/signal/'
noise_path = './data/noise/'
batch_size = 4
image_size = 224

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

image = CustomImageDataset(config['image_path'])
print(image)

signal = CustomSignalDataset(config['signal_path'])
print(signal.shape)

noise = CustomNoiseDataset(config['noise_path'])
print(noise.shape)

data = {'image':image, 'signal':signal, 'noise':noise}

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

    def preprocess(self):
        image = self.data['image']
        resized_image = image.resize(config['image_size'])
        image_array = np.array(resized_image)
        normalized_image_array = image_array / 255.0

        return normalized_image_array