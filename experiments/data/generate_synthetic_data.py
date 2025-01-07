import numpy as np
import torch
from torch import nn

######### nowhere near done hear, quick pause###################
def generate_shifted_quantized_sine_waves(seq_length, num_samples, num_classes, noise_factor=0.05):
    x = np.linspace(0, 4*np.pi, seq_length)
    # Generate 1000 different shifts
    shifts = np.random.uniform(0, 2*np.pi, num_samples)

    # Initialize an array to store all shifted sine waves (including noisy versions)
    all_waves = np.zeros((num_samples * 11, seq_length))

    for i, shift in enumerate(shifts):
        y = np.sin(x + shift) * np.cos(x * shift)

        # Normalize to [0, 1] range
        y = (y + 1) / 2

        # Generate the original quantized wave
        y_quantized = np.floor(y * num_classes).astype(int)
        all_waves[i * 11] = y_quantized

        # Generate 10 noisy versions of the wave
        for j in range(1, 11):
            # Add noise to the original continuous wave
            y_noisy = y + np.random.normal(0, noise_factor, seq_length)

            # Clip values to [0, 1] range
            y_noisy = np.clip(y_noisy, 0, 1)

            # Quantize the noisy wave
            y_noisy_quantized = np.floor(y_noisy * num_classes).astype(int)

            # Store the noisy quantized wave
            all_waves[i * 11 + j] = y_noisy_quantized

    return all_waves
