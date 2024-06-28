import os
import random
import pandas as pd

image_path = None
signal_path = None
noise_path = None

config = {
    image_path,
    signal_path,
    noise_path,
}

img_list = os.walk(config['image_path'])
noise_list = os.walk(config['noise_path'])