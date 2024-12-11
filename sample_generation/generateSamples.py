import os
import numpy as np
from funcSampleGeneration import generate_unlabeled_modality_images
from funcLabelSampleGeneration import generate_labeled_modality_images
import shutil
from tqdm import tqdm
from typing import List, Tuple

def generate_modalities(
    samples_per_image: int,
    image_size: Tuple[int, int],
    image_num: List[int],
    mod_types: List[str],
    set_types: List[str],
    label: bool,
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
    label (bool): If True, use 'labeled', otherwise 'unlabeled'.
    mode (str): 'train' or 'test'
    base_path (str): Base directory to store generated data
    """

    label_type = 'labeled' if label else 'unlabeled'
    fold_path = os.path.join(base_path, label_type, mode)
    
    # Create necessary directories
    for gen_type in set_types:
        os.makedirs(os.path.join(fold_path, gen_type), exist_ok=True)

    # Determine the function to call based on the label
    generate_images_function = (
        generate_labeled_modality_images if label else generate_unlabeled_modality_images
    )

    # Generate images for each modulation type
    for mod in tqdm(mod_types, desc="Generating images for modalities"):
        generate_images_function(
            mod, samples_per_image, image_num[0], image_size, set_types, fold_path
        )
    
    print("Processing complete.")

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'samples_per_image': 1024,
        'image_size': (224, 224),
        'image_num': [10], # total number of images to generate: image_num * len(mod_types)
        'mod_types': ['OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 'GFSK', '4PAM', 'DQPSK', '16PAM', 'GMSK'],
        'set_types': ['noiseLessImg', 'noisyImg', 'noiselessSignal', 'noise', 'noisySignal'],
        'label': False,
        'mode': 'test',
        'base_path': '/mnt/d/OneDrive - Rowan University/RA/Fall 24/DenoMAE_2.0/sample_generation/data'
    }

    generate_modalities(**CONFIG)
