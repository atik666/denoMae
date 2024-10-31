import os
import numpy as np
from funcSampleGeneration import generate_constellation_images
import shutil
from tqdm import tqdm
from typing import List, Tuple

def generate_constellations(
    samples_per_image: int,
    image_size: Tuple[int, int],
    image_num: List[int],
    mod_types: List[str],
    set_types: List[str],
    mode: str,
    base_path: str
) -> None:
    """
    Generate constellation images for various modulation types.
    
    Args:
    samples_per_image (int): Number of samples to produce each constellation image
    image_size (Tuple[int, int]): Size of the output images
    image_num (List[int]): Number of images to generate per modulation type
    mod_types (List[str]): List of modulation types
    set_types (List[str]): Types of data to generate (e.g., noiseless, noisy)
    mode (str): 'train' or 'test'
    base_path (str): Base directory to store generated data
    """
    fold_path = os.path.join(base_path, 'unlabeled', mode)
    
    # Create necessary directories
    for gen_type in set_types:
        os.makedirs(os.path.join(fold_path, gen_type), exist_ok=True)

    # Generate images for each modulation type
    for mod in tqdm(mod_types, desc="Generating images for modalities"):
        generate_constellation_images(
            mod, samples_per_image, image_num[0], image_size, set_types, fold_path
        )
    
    print("Processing complete.")

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'samples_per_image': 1024,
        'image_size': (224, 224),
        'image_num': [5], # total number of images to generate: image_num * len(mod_types)
        'mod_types': ['OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 'GFSK', '4PAM', 'DQPSK', '16PAM', 'GMSK'],
        'set_types': ['noiseLessImg', 'noisyImg', 'noiselessSignal', 'noise', 'noisySignal'],
        'mode': 'test_out',
        # 'base_path': './data'
        'base_path': '/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data'
    }

    generate_constellations(**CONFIG)
