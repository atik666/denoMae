import os
import numpy as np
from scipy.constants import pi
from scipy.signal import convolve
from numpy.random import randn
from PIL import Image
from datetime import datetime
from typing import Dict, Tuple, List
from functools import partial
from numpy.random import randn, randint

# AWGN function
def awgn(signal: np.ndarray, snr_dB: float) -> Tuple[np.ndarray, np.ndarray]:
    snr = 10**(snr_dB / 10.0)
    power_signal = np.mean(np.abs(signal)**2)
    power_noise = power_signal / snr
    noise = np.sqrt(power_noise / 2) * (randn(signal.size) + 1j * randn(signal.size))
    return signal + noise, noise

# GMSK modulation function
def gmsk_modulate(bits: np.ndarray, bt_product: float, samples_per_symbol: int) -> np.ndarray:
    h = 0.5
    g = np.sinc(np.arange(-4, 4 + 1 / samples_per_symbol, 1 / samples_per_symbol)) * np.hamming(8 * samples_per_symbol + 1)
    g /= np.sum(g)
    freq = convolve(np.repeat(bits, samples_per_symbol), g, mode='same')
    phase = np.cumsum(freq * pi * h)
    return np.exp(1j * phase)

# image generation function
def generate_image(signal: np.ndarray, imageSize: Tuple[int, int], imageDir: str, imageName: str) -> None:
    
    """Generate constellation image from signal."""

    # TODO: need to figure out these parameters
    blkSize = [5, 25, 50]
    cFactor = 5.0 / np.array(blkSize)

    # cons_scale (List[float]): Scale of constellation image
    consScale = [2.5, 2.5]

    maxBlkSize = max(blkSize)
    imageSizeX, imageSizeY = imageSize[0] + 4 * maxBlkSize, imageSize[1] + 4 * maxBlkSize
    consScaleI, consScaleQ = consScale[0] + 2 * maxBlkSize * (2 * consScale[0] / imageSize[0]), consScale[1] + 2 * maxBlkSize * (2 * consScale[1] / imageSize[1])
    
    dIY, dQX = 2 * consScale[0] / imageSize[0], 2 * consScale[1] / imageSize[1]
    dXY = np.sqrt(dIY**2 + dQX**2)

    # Calculate sample positions
    sampleX = np.rint((consScaleQ - np.imag(signal)) / dQX).astype(int)
    sampleY = np.rint((consScaleI + np.real(signal)) / dIY).astype(int)

    # Create pixel centroid grid
    ii, jj = np.meshgrid(range(imageSizeX), range(imageSizeY), indexing='ij')
    pixelCentroid = (-consScaleI + dIY / 2 + jj * dIY) + 1j * (consScaleQ - dQX / 2 - ii * dQX)

    imageArray = np.zeros((imageSizeX, imageSizeY, 3))

    for kk, blk in enumerate(blkSize):
        blkXmin, blkXmax = sampleX - blk, sampleX + blk + 1
        blkYmin, blkYmax = sampleY - blk, sampleY + blk + 1
        
        valid = (blkXmin > 0) & (blkYmin > 0) & (blkXmax < imageSizeX) & (blkYmax < imageSizeY)
        
        for ii in np.where(valid)[0]:
            sampleDistance = np.abs(signal[ii] - pixelCentroid[blkXmin[ii]:blkXmax[ii], blkYmin[ii]:blkYmax[ii]])
            imageArray[blkXmin[ii]:blkXmax[ii], blkYmin[ii]:blkYmax[ii], kk] += np.exp(-cFactor[kk] * sampleDistance / dXY)

        imageArray[:, :, kk] /= np.max(imageArray[:, :, kk])

    imageArray = (imageArray * 255).astype(np.uint8)
    im = Image.fromarray(imageArray[2*maxBlkSize:-2*maxBlkSize, 2*maxBlkSize:-2*maxBlkSize])
    im.save(os.path.join(imageDir, f"{imageName}.png"))

# Main function to generate constellation images
def generate_constellation_images(modType: str, samplesPerImage: int, imageNum: int, imageSize: Tuple[int, int], set_type: List[str], setPath: str) -> None:
    """Generate constellation images for various modulation types."""

    ul, ll = -11, -20
    SNR_dB = np.random.uniform(ll, ul)

    # Define modulation types and their corresponding constellation diagrams
    mod_types: Dict[str, Tuple[np.ndarray, int]] = {
        'OOK': (np.array([0, 1]), 1),
        '4ASK': (np.array([-3, -1, 1, 3]), 2),
        '8ASK': (np.array([-7, -5, -3, -1, 1, 3, 5, 7]), 3),
        'OQPSK': (np.exp((np.arange(4) / 4) * 2 * np.pi * 1j + np.pi / 4), 2),
        'CPFSK': (np.exp(1j * 2 * np.pi * np.array([0.25, 0.75])), 1),
        'GFSK': (np.exp(1j * 2 * np.pi * np.array([0.25, 0.75])), 1),
        '4PAM': (np.array([-3, -1, 1, 3]), 2),
        'DQPSK': (np.exp(1j * np.array([0, np.pi/2, np.pi, -np.pi/2])), 2),
        '16PAM': (np.arange(-15, 16, 2), 4),
        'GMSK': (np.array([0, 1]), 1)  # GMSK uses binary symbols
    }

    if modType not in mod_types:
        raise ValueError('Unrecognized Modulation Type!')
    
    consDiag, modOrder = mod_types[modType]

    btProduct = 0.3 if modType == 'GMSK' else None
    samplesPerSymbol = 8 if modType == 'GMSK' else None

    imageIDPrefix = f"{modType}_{SNR_dB:.2f}dB__"

    # Create directories for each set type
    image_dirs = {genType: os.path.join(setPath, genType) for genType in set_type}
    for dir_path in image_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    generate_image_partial = partial(generate_image, imageSize=imageSize)

    for jj in range(imageNum):
        if modType == 'GMSK':
            msgBits = randint(0, 2, samplesPerImage)
            signalTx = gmsk_modulate(msgBits, btProduct, samplesPerSymbol)
        else:
            msg = randint(len(consDiag), size=samplesPerImage)
            signalTx = consDiag[msg]

        signalTx = signalTx.astype(np.complex128)
            
        if modType in ['BPSK', '4ASK']:
            signalTx[0] += 1j * 1E-4
        
        imageID = f"{jj:0{len(str(imageNum))}d}"
        imageName = f"{imageIDPrefix}{imageID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        phaseOffset = np.random.normal(0, 0.0001, len(signalTx)) + np.arange(len(signalTx)) * np.random.normal(0, 0.0001)
        signalTx *= np.exp(1j * phaseOffset)

        signalRx, noise = awgn(signalTx, SNR_dB) # TODO: the SNR stays the same for all images in one class. Make it different for each image. It is important and correct it.

        # Generate images and save signals for each set type
        for genType in set_type:
            if genType == 'noiseLessImg':
                generate_image_partial(signal=signalTx, imageDir=image_dirs[genType], imageName=imageName)
            elif genType == 'noisyImg':
                generate_image_partial(signal=signalRx, imageDir=image_dirs[genType], imageName=imageName)
            elif genType in ['noiselessSignal', 'noise', 'noisySignal']:
                signal_to_save = {'noiselessSignal': signalTx, 'noise': noise, 'noisySignal': signalRx}[genType]
                np.save(os.path.join(image_dirs[genType], f"{imageName}.npy"), signal_to_save)

        # Log generated files
        with open(os.path.join(setPath, "files.txt"), 'a') as file:
            file.write(f"{imageName}\n")