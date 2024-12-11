import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate a sample 1D signal
np.random.seed(0)
signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1024)) + np.random.normal(0, 0.5, 1024)

# Define wavelet and scales for CWT
wavelet = 'cmor'  # Complex Morlet wavelet
scales = np.arange(1, 128)

# Compute the CWT
coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/1024)

# Create a spectrogram plot
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='viridis', aspect='auto', origin='lower')
plt.axis('off')  # Remove axis titles and labels
plt.show()
