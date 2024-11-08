# %%
"""### Utils

"""

from typing import Dict, Optional
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from tqdm import tqdm
from pathlib import Path
import datasets
import pandas as pd
from collections import defaultdict
from itertools import product
import torch
import matplotlib.colors as colors

from pytorch_model_summary import summary

# using a faster style for plotting
mplstyle.use('fast')

# Mapping of nucleotides to float coordinates
mapping_easy = {
    'A': np.array([0.5, -0.8660254037844386]),
    'T': np.array([0.5, 0.8660254037844386]),
    'G': np.array([0.8660254037844386, -0.5]),
    'C': np.array([0.8660254037844386, 0.5]),
    'N': np.array([0, 0])
}


# %%
# get datasets
viral_ref = datasets.load_dataset('Hack90/ref_seq_viral')
## df for ease of use
viral_ref = pd.DataFrame(viral_ref['train'])


aquatic_bird_bornavirus_full = viral_ref['sequence'].iloc[18717]
aquatic_bird_bornavirus_medium = aquatic_bird_bornavirus_full[3000:4000]
aquatic_bird_bornavirus_small = aquatic_bird_bornavirus_full[3500:3600]


# %%
# coordinates for x+iy
Coord = namedtuple("Coord", ["x","y"])

# coordinates for a CGR encoding
CGRCoords = namedtuple("CGRCoords", ["N","x","y"])

# coordinates for each nucleotide in the 2d-plane
DEFAULT_COORDS = dict(A=Coord(1,1),C=Coord(-1,1),G=Coord(-1,-1),T=Coord(1,-1))

# Function to convert a DNA sequence to a list of coordinates
def _dna_to_coordinates(dna_sequence, mapping):
    dna_sequence = dna_sequence.upper()
    coordinates = np.array([mapping.get(nucleotide, mapping['N']) for nucleotide in dna_sequence])
    return coordinates

# Function to create the cumulative sum of a list of coordinates
def _get_cumulative_coords(mapped_coords):
    cumulative_coords = np.cumsum(mapped_coords, axis=0)
    return cumulative_coords

# Function to take a list of DNA sequences and plot them in a single figure
def plot_2d_sequences(dna_sequences, mapping=mapping_easy, single_sequence=True):
    fig, ax = plt.subplots()
    if single_sequence:
        dna_sequences = [dna_sequences]
    for dna_sequence in dna_sequences:
        mapped_coords = _dna_to_coordinates(dna_sequence, mapping)
        cumulative_coords = _get_cumulative_coords(mapped_coords)
        ax.plot(*cumulative_coords.T)
    ## plt.show()

# Function to take a list of DNA sequences and plot them in a single figure
def plot_2d_sequences_returnfig(dna_sequences, mapping=mapping_easy, single_sequence=True):
    fig, ax = plt.subplots()
    if single_sequence:
        dna_sequences = [dna_sequences]
    for dna_sequence in dna_sequences:
        mapped_coords = _dna_to_coordinates(dna_sequence, mapping)
        cumulative_coords = _get_cumulative_coords(mapped_coords)
        ax.plot(*cumulative_coords.T)
    return fig, ax

# Function to take a list of DNA sequences and plot them in a single figure
def return_2d_sequences(dna_sequences, mapping=mapping_easy, single_sequence=True):
    if single_sequence:
        dna_sequences = [dna_sequences]
    for dna_sequence in dna_sequences:
        mapped_coords = _dna_to_coordinates(dna_sequence, mapping)
        cumulative_coords = _get_cumulative_coords(mapped_coords)
        return cumulative_coords.T

## Setting up training Data
def create_train_test_set(seq):
  representation = return_2d_sequences(seq)
  xy_pairs = np.column_stack((representation[0], representation[1]))
  xy_tensor = torch.tensor(xy_pairs)
  train_ratio = 0.9
  train_size = int(train_ratio * len(xy_tensor))
  indices = torch.randperm(len(xy_tensor))
  train_indices = indices[:train_size]
  test_indices = indices[train_size:]
  train_set = xy_tensor[train_indices]
  test_set = xy_tensor[test_indices]

  train_labels = train_set[:,1:]
  train_inputs = train_set[:,:1]
  test_labels = test_set[:,1:]
  test_inputs = test_set[:,:1]

  return train_labels, train_inputs, test_labels, test_inputs


import pywt
import numpy as np
import matplotlib.pyplot as plt

def lowpassfilter(signal, thresh=0.95, wavelet="coif5", clip_thresh=50):
    thresh = thresh * np.nanmax(signal)

    coeff = pywt.wavedec(signal, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    # Set the reconstructed signal to NaN where it deviates too much from the original
    deviation = np.abs(reconstructed_signal - signal)
    reconstructed_signal = np.where(deviation > clip_thresh, np.nan, reconstructed_signal)
    return reconstructed_signal

## Setting up training Data
def create_train_test_set_filtered(seq):
  representation = return_2d_sequences(seq)
  representation[1] = lowpassfilter(representation[1])
  xy_pairs = np.column_stack((representation[0], representation[1]))
  xy_tensor = torch.tensor(xy_pairs)
  train_ratio = 0.9
  train_size = int(train_ratio * len(xy_tensor))
  indices = torch.randperm(len(xy_tensor))
  train_indices = indices[:train_size]
  test_indices = indices[train_size:]
  train_set = xy_tensor[train_indices]
  test_set = xy_tensor[test_indices]

  train_labels = train_set[:,1:]
  train_inputs = train_set[:,:1]
  test_labels = test_set[:,1:]
  test_inputs = test_set[:,:1]

  return train_labels, train_inputs, test_labels, test_inputs

import torch
logits = []

melting_stability = {
    "TA": -0.12,
    "TG": -0.78, "TC": -0.78,
    "CG": -1.44,
    "AG": -1.29, "CT": -1.29,
    "AA": -1.04, "TT": -1.04,
    "AT": -1.27,
    "GA": -1.66, "TC": -1.66,
    "CC": -1.97, "GG": -1.97,
    "AC": -2.04, "GT": -2.04,
    "GC": -2.70,
    "N": 0  # Default value for unspecified pairs
}

# Adjust the function to include the melting stability scaling
def _dna_to_coordinates_scaled(dna_sequence, mapping, stability):
    dna_sequence = dna_sequence.upper()
    # Initialize the list to store scaled coordinates
    coordinates = []
    for i in range(len(dna_sequence) - 1):
        nucleotide_pair = dna_sequence[i:i+2]
        # Determine the movement vector and scale it by the melting stability factor
        base_movement = mapping.get(dna_sequence[i+1], mapping['N'])
        scale = -stability.get(nucleotide_pair, stability['N'])
        scaled_movement = base_movement * abs(scale)
        coordinates.append(scaled_movement)
    return np.array(coordinates)

# Adjusted function to plot sequences
def plot_2d_sequences_scaled(dna_sequences, mapping=mapping_easy, stability=melting_stability, single_sequence=True):
    fig, ax = plt.subplots()
    if single_sequence:
        dna_sequences = [dna_sequences]
    for dna_sequence in dna_sequences:
        mapped_coords = _dna_to_coordinates_scaled(dna_sequence, mapping, stability)
        cumulative_coords = _get_cumulative_coords(mapped_coords)
        ax.plot(*cumulative_coords.T)
    ## plt.show()

# Function to return coordinates of sequences
def return_2d_sequences_scaled(dna_sequences, mapping=mapping_easy, stability=melting_stability, single_sequence=True):
    if single_sequence:
        dna_sequences = [dna_sequences]
    for dna_sequence in dna_sequences:
        mapped_coords = _dna_to_coordinates_scaled(dna_sequence, mapping, stability)
        cumulative_coords = _get_cumulative_coords(mapped_coords)
        return cumulative_coords.T

# Example usage:
# plot_2d_sequences_scaled(aquatic_bird_bornavirus_small)

# #plot_2d_sequences(viral_ref['sequence'].to_list()[4590])
# coords = return_2d_sequences_scaled(aquatic_bird_bornavirus_medium)
# print(coords)
# %%

# for fig 1 2d reps

plot_2d_sequences(aquatic_bird_bornavirus_full) # full sequence
plt.savefig("outputs/aquatic_bird_bornavirus_full.png", format='png')

plot_2d_sequences(aquatic_bird_bornavirus_medium) # 1000 bp, starting at point 3000
plt.savefig("outputs/aquatic_bird_bornavirus_medium.png", format='png')

plot_2d_sequences(aquatic_bird_bornavirus_small) # 100, even as we get smaller/closer, chaotic nature persits. Fancy graph to be made
# overlaying each
plt.savefig("outputs/aquatic_bird_bornavirus_small.png", format='png')

plot_2d_sequences_scaled(aquatic_bird_bornavirus_full) # full sequence
plt.savefig("outputs/aquatic_bird_bornavirus_scaled_full.png", format='png')

plot_2d_sequences_scaled(aquatic_bird_bornavirus_medium) # 1000 bp, starting at point 3000
plt.savefig("outputs/aquatic_bird_bornavirus_scaled_medium.png", format='png')

plot_2d_sequences_scaled(aquatic_bird_bornavirus_small) # 100, even as we get smaller/closer, chaotic nature persits. Fancy graph to be made
# overlaying each
plt.savefig("outputs/aquatic_bird_bornavirus_scaled_small.png", format='png')


# %%

import numpy as np
from scipy.stats import linregress

def adaptive_fractal_analysis(x, y, n_min=5, n_max=100, n_steps=20):
    """
    Perform Adaptive Fractal Analysis (AFA) on a time series.

    Args:
        x (array-like): The x-coordinates of the time series.
        y (array-like): The y-coordinates of the time series.
        n_min (int): The minimum window size. Default is 5.
        n_max (int): The maximum window size. Default is 100.
        n_steps (int): The number of window sizes to consider between n_min and n_max. Default is 20.

    Returns:
        float: The estimated Hurst exponent.
    """
    # Sort the data points by x-coordinate
    sorted_indices = np.argsort(x)
    x = np.array(x)[sorted_indices]
    y = np.array(y)[sorted_indices]

    window_sizes = np.logspace(np.log10(n_min), np.log10(n_max), n_steps).astype(int)
    fluctuations = []

    for w in window_sizes:
        # Partition the data into overlapping windows of size w
        windows = [y[i:i+w] for i in range(len(y) - w + 1)]

        # Compute the local linear approximations for each window
        approx = np.zeros_like(y)
        for i, window in enumerate(windows):
            # Perform linear regression on the window
            slope, intercept, _, _, _ = linregress(np.arange(w), window)
            approx[i:i+w] += intercept + slope * np.arange(w)

        # Compute the fluctuation function F(w)
        fluctuation = np.sqrt(np.mean((y - approx)**2))
        fluctuations.append(fluctuation)

    # Perform linear regression on log(F(w)) vs log(w)
    slope, _, _, _, _ = linregress(np.log(window_sizes), np.log(fluctuations))

    # Estimate the Hurst exponent from the slope of the regression line
    hurst_exponent = slope

    return hurst_exponent

x = return_2d_sequences(aquatic_bird_bornavirus_full)[0].flatten() # Your x-coordinates
y = return_2d_sequences(aquatic_bird_bornavirus_full)[1].flatten()  # Your y-coordinates

hurst_exponent = adaptive_fractal_analysis(x, y)
print(f"Estimated Hurst exponent: {hurst_exponent:.3f}")

import numpy as np
import matplotlib.pyplot as plt

def afa(x, y, w):
    T = len(y)
    n = (w - 1) // 2
    y_approx = np.zeros(T)

    for i in range(n, T-n):
        xi = np.arange(i-n, i+n+1)
        yi = y[i-n:i+n+1]
        A = np.vstack([xi, np.ones(len(xi))]).T
        m, c = np.linalg.lstsq(A, yi, rcond=None)[0]
        y_approx[i] = m * i + c

    return y_approx

def F_w(y, y_approx, T):
    diff = y - y_approx
    return np.sqrt((1/T) * np.sum(diff**2))

def estimate_hurst_exponent(x, y, window_sizes):
    T = len(y)
    F_w_values = []

    for w in window_sizes:
        y_approx = afa(x, y, w)
        F_w_value = F_w(y, y_approx, T)
        F_w_values.append(F_w_value)

    log_w = np.log(window_sizes)
    log_F_w = np.log(F_w_values)

    poly = np.polyfit(log_w, log_F_w, 1)
    H = poly[0]

    return H, log_w, log_F_w

# Example dataset
x = return_2d_sequences_scaled(aquatic_bird_bornavirus_full)[0].flatten() # Your x-coordinates
y = return_2d_sequences_scaled(aquatic_bird_bornavirus_full)[1].flatten()

# Define window sizes
window_sizes = [4,5,7,9,14,70, 11, 21, 51, 101, 1000, 300, 3000]

# Estimate the Hurst exponent
H, log_w, log_F_w = estimate_hurst_exponent(x, y, window_sizes)

# %%
# Plot the results
plt.figure(figsize=(7, 5))
plt.plot(log_w, log_F_w, 'o-', label='Data')
plt.plot(log_w, np.polyval(np.polyfit(log_w, log_F_w, 1), log_w), 'r--', label=f'Fit (H={H:.3f})')
plt.xlabel('log(w)')
plt.ylabel('log(F(w))')
plt.legend()
plt.title('Estimated Hurst Exponent using AFA')
#plt.show()
plt.savefig("outputs/hurst-exponent.png", format='png')

print(f"Estimated Hurst Exponent: H = {H:.3f}")
# %%
import numpy as np

def afa(x, y, w):
    T = len(y)
    n = (w - 1) // 2
    y_approx = np.zeros(T)
    for i in range(n, T-n):
        xi = np.arange(i-n, i+n+1)
        yi = y[i-n:i+n+1]
        A = np.vstack([xi, np.ones(len(xi))]).T
        m, c = np.linalg.lstsq(A, yi, rcond=None)[0]
        y_approx[i] = m * i + c
    return y_approx

def F_w(y, y_approx, T):
    diff = y - y_approx
    return np.sqrt((1/T) * np.sum(diff**2))

def estimate_hurst_exponent(sequence):
    x = return_2d_sequences(sequence)[0].flatten()  # Your x-coordinates
    y = return_2d_sequences(sequence)[1].flatten()  # Your y-coordinates
    T = len(y)

    window_sizes = [4, 5, 7, 9, 14, 70, 11, 21, 51, 101, 1000, 300]
    F_w_values = []

    for w in window_sizes:
        y_approx = afa(x, y, w)
        F_w_value = F_w(y, y_approx, T)
        F_w_values.append(F_w_value)

    log_w = np.log(window_sizes)
    log_F_w = np.log(F_w_values)
    poly = np.polyfit(log_w, log_F_w, 1)
    H = poly[0]

    return H

def estimate_hurst_exponent_scaled(sequence):
    x = return_2d_sequences_scaled(sequence)[0].flatten()  # Your x-coordinates
    y = return_2d_sequences_scaled(sequence)[1].flatten()  # Your y-coordinates
    T = len(y)

    window_sizes = [4, 5, 7, 9, 14, 70, 11, 21, 51, 101, 1000, 300]
    F_w_values = []

    for w in window_sizes:
        y_approx = afa(x, y, w)
        F_w_value = F_w(y, y_approx, T)
        F_w_values.append(F_w_value)

    log_w = np.log(window_sizes)
    log_F_w = np.log(F_w_values)
    poly = np.polyfit(log_w, log_F_w, 1)
    H = poly[0]

    return H

# Example usage
  # Your list of sequences
# sample
vir_samp = viral_ref['sequence'].sample(100)

hurst_exponents = []
for sequence in vir_samp.to_list():
    H = estimate_hurst_exponent(sequence)
    hurst_exponents.append(H)
    print(f"Estimated Hurst Exponent for sequence: H = {H:.3f}")
# %%
# Plot the distribution of Hurst exponents
plt.figure(figsize=(7, 5))
plt.hist(hurst_exponents, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Hurst Exponent')
plt.ylabel('Frequency')
plt.title('Distribution of Hurst Exponents')
plt.grid(True)
## plt.show()
plt.savefig("outputs/hurst-exp-distr.png", format='png')

# %%
hurst_exponents = []
for sequence in vir_samp.to_list():
    H = estimate_hurst_exponent_scaled(sequence)
    hurst_exponents.append(H)
    print(f"Estimated Hurst Exponent for sequence: H = {H:.3f}")
# %%
# Plot the distribution of Hurst exponents
plt.figure(figsize=(7, 5))
plt.hist(hurst_exponents, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Hurst Exponent')
plt.ylabel('Frequency')
plt.title('Distribution of Hurst Exponents')
plt.grid(True)
## plt.show()
plt.savefig("outputs/hurst-exp-distr-scaled.png", format='png')

# %%
# viral_ref # something about 18k species and plot the species families plot

"""##### Fractal Nature of DNA

"""

train_labels, train_inputs, test_labels, test_inputs = create_train_test_set(aquatic_bird_bornavirus_small)

train_labels, train_inputs, test_labels, test_inputs = create_train_test_set_filtered(aquatic_bird_bornavirus_full)

 # %%
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.hidden_layers.append(nn.GELU())
        self.hidden_layers.append(nn.Dropout(dropout_rate))  # Added dropout after GELU
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.hidden_layers.append(nn.GELU())
            self.hidden_layers.append(nn.Dropout(dropout_rate))  # Added dropout after GELU
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.output_layer(x)
        return out

# Normalize the labels (y) using StandardScaler
scaler = StandardScaler()
train_labels = scaler.fit_transform(train_labels.reshape(-1, 1))
test_labels = scaler.transform(test_labels.reshape(-1, 1))

train_inputs = scaler.fit_transform(train_inputs.reshape(-1, 1))
test_inputs = scaler.transform(test_inputs.reshape(-1, 1))

# Convert data to PyTorch tensors
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Define the model and optimizer

input_size = train_inputs.shape[1]
hidden_sizes = [128, 256, 512, 256, 128]#]  # Example hidden layer sizes
output_size = 1

model = MLP(input_size, hidden_sizes, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)  # Added L2 regularization


# Train the model
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(train_inputs), batch_size):
        batch_inputs = train_inputs[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]

        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the test set
with torch.no_grad():
    test_outputs = model(test_inputs)
    test_outputs = scaler.inverse_transform(test_outputs.numpy())
    test_loss = criterion(torch.tensor(test_outputs), test_labels)
    print(f"Test Loss: {test_loss.item():.4f}")

# Generate predictions for a range of input values
input_range = np.linspace(train_inputs.min().item(), train_inputs.max().item(), 100)
input_range = torch.tensor(input_range, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    predicted_range = model(input_range)
    predicted_range = scaler.inverse_transform(predicted_range.numpy())

# Plot the original data and the learned function
plt.figure(figsize=(8, 6))
plt.scatter(scaler.inverse_transform(train_inputs.numpy()), scaler.inverse_transform(train_labels.numpy()), label='Training Data')
plt.scatter(scaler.inverse_transform(test_inputs.numpy()), scaler.inverse_transform(test_labels.numpy()), label='Test Data')
plt.plot(input_range.numpy(), predicted_range, color='red', label='Learned Function')

# Plot the original data
original_inputs = np.concatenate((train_inputs.numpy(), test_inputs.numpy()))
original_labels = np.concatenate((scaler.inverse_transform(train_labels.numpy()), scaler.inverse_transform(test_labels.numpy())))
#plt.scatter(original_inputs, original_labels, color='green', label='Original Data')

plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Learned Function and Original Data')
plt.legend()
#plt.show()
plt.savefig("outputs/learned-function-orig-data.png", format='png')

# %%

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


input_size = 1
hidden_sizes = [32, 64, 128, 64, 32]
output_size = 1
learning_rate = 0.01
num_epochs = 1000
step_size = 200
gamma = 0.1

# Create the MLP model
model = MLP(input_size, hidden_sizes, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training data (example)
x_train = torch.tensor(return_2d_sequences(aquatic_bird_bornavirus_small)[0]).float().view(-1, 1)
y_train = torch.tensor(return_2d_sequences(aquatic_bird_bornavirus_small)[1]).float().view(-1, 1)

# Scale the input data to a smaller range
x_train_scaled = (x_train - x_train.mean()) / x_train.std()
y_train_scaled = (y_train - y_train.mean()) / y_train.std()

# Store the predicted values
y_pred_scaled = []

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train_scaled)
    loss = criterion(outputs, y_train_scaled)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    # Store the predicted values
    y_pred_scaled.append(outputs.detach().numpy())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, lr: {scheduler.get_last_lr()[0]:.6f}')

# Convert PyTorch tensors to NumPy arrays
y_train_np = y_train.numpy().flatten()
y_train_std_np = y_train.std().item()
y_train_mean_np = y_train.mean().item()

# Unscale the predicted values
y_pred = [y_pred_scaled[i] * y_train_std_np + y_train_mean_np for i in range(len(y_pred_scaled))]

# Plot the original and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(x_train.numpy(), y_train_np, label='Original')
plt.scatter(x_train.numpy(), y_pred[-1], label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Original vs. Predicted Values')
#plt.show()
plt.savefig("outputs/orig-vs-pred-values-2drep.png", format='png')

# %%

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def evaluate_sequence(sequence, input_size=1, hidden_sizes=[32, 64, 128, 64, 32], output_size=1,
                      learning_rate=0.01, num_epochs=15, step_size=200, gamma=0.1):

    # Extract input and target values from the sequence
    x_train = torch.tensor(return_2d_sequences(sequence)[0]).float().view(-1, 1)
    y_train = torch.tensor(return_2d_sequences(sequence)[1]).float().view(-1, 1)

    # Scale the input and target data
    x_train_scaled = (x_train - x_train.mean()) / x_train.std()
    y_train_scaled = (y_train - y_train.mean()) / y_train.std()

    # Create the MLP model
    model = MLP(input_size, hidden_sizes, output_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x_train_scaled)
        loss = criterion(outputs, y_train_scaled)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, lr: {scheduler.get_last_lr()[0]:.6f}')

    # Calculate the final MSE on the training data
    final_output = model(x_train_scaled)
    mse = criterion(final_output, y_train_scaled).item()

    return mse, model

# Example usage:
sequence_list = [aquatic_bird_bornavirus_full, aquatic_bird_bornavirus_medium, aquatic_bird_bornavirus_small] # Replace with actual sequences
mse_list = []

for seq in sequence_list:
    mse, model = evaluate_sequence(seq)
    mse_list.append(mse)
    print(f'MSE for the given sequence: {mse:.6f}')

print("MSE list:", mse_list)

 # %%
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def evaluate_sequence(sequence, percentages, input_size=1, hidden_sizes=[32, 64, 128, 64, 32], output_size=1,
                      learning_rate=0.01, num_epochs=1000, step_size=200, gamma=0.1):

    # Extract input and target values from the sequence
    x_train = torch.tensor(return_2d_sequences(sequence)[0]).float().view(-1, 1)
    y_train = torch.tensor(return_2d_sequences(sequence)[1]).float().view(-1, 1)

    # Normalize the full dataset
    x_train_scaled = (x_train - x_train.mean()) / x_train.std()
    y_train_scaled = (y_train - y_train.mean()) / y_train.std()

    all_mse = []

    for percentage in percentages:
        mse_list = []
        num_samples = int(len(x_train) * percentage)

        for _ in range(10):  # Sample 10 times
            # Sample data points
            x_sample, y_sample = resample(x_train_scaled, y_train_scaled, n_samples=num_samples)

            # Create the MLP model
            model = MLP(input_size, hidden_sizes, output_size)

            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Learning rate scheduler
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

            # Training loop
            for epoch in range(num_epochs):
                # Forward pass
                outputs = model(x_sample)
                loss = criterion(outputs, y_sample)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the learning rate
                scheduler.step()

            # Calculate the MSE on the sampled training data
            final_output = model(x_train_scaled)
            mse = criterion(final_output, y_train_scaled).item()
            mse_list.append(mse)

        # Store the average MSE for this percentage
        mean_mse = np.mean(mse_list)
        all_mse.append(mean_mse)
        print(f'Percentage: {percentage:.2f}, Average MSE: {mean_mse:.6f}')

    return percentages, all_mse

# Define your sequence and percentages
sequence = aquatic_bird_bornavirus_full  # Replace with actual sequence
percentages = [0.1, 0.2, 0.5, 0.7, 1.0]  # Define your percentages here

# Call the function
percentages, mse_list = evaluate_sequence(sequence, percentages)

# Create a DataFrame to show the results
results_df = pd.DataFrame({
    'Percentage': percentages,
    'Average MSE': mse_list
})

# Print the table
print(results_df)

# Plot the Percentage vs. Average MSE
plt.figure(figsize=(10, 6))
plt.plot(percentages, mse_list, marker='o', linestyle='-', color='blue')
plt.xlabel('Percentage of Data')
plt.ylabel('Average MSE')
plt.title('Percentage of Data vs. Average MSE')
plt.grid(True)
#plt.show()
plt.savefig("outputs/perc-data-vs-context-win.png", format='png')

# %%

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



def evaluate_sequence(sequence, percentages, input_size=1, hidden_sizes=[32, 64, 128, 64, 32], output_size=1,
                      learning_rate=0.01, num_epochs=10, step_size=200, gamma=0.1):

    all_mse = []

    for percentage in percentages:
        mse_list = []
        k = int(len(sequence) * percentage)

        if k < 1:
            print(f"Skipping percentage {percentage} as it results in length < 1")
            continue

        for _ in range(25):  # Sample 10 times
            start_idx = np.random.randint(0, len(sequence) - k + 1)
            sample_seq = sequence[start_idx:start_idx + k]

            # Extract input and target values from the sequence
           # print(return_2d_sequences(sample_seq)[1])
            x_train = torch.tensor(return_2d_sequences(sample_seq)[0]).float().view(-1, 1)
            y_train = torch.tensor(return_2d_sequences(sample_seq)[1]).float().view(-1, 1)

            # Scale the input and target data
            x_train_scaled = (x_train - x_train.mean()) / x_train.std()
            y_train_scaled = (y_train - y_train.mean()) / y_train.std()

            # Create the MLP model
            model = MLP(input_size, hidden_sizes, output_size)

            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Learning rate scheduler
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

            # Training loop
            for epoch in range(num_epochs):
                # Forward pass
                outputs = model(x_train_scaled)
                loss = criterion(outputs, y_train_scaled)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the learning rate
                scheduler.step()

            # Calculate the MSE on the full sequence (unseen data)
            x_full = torch.tensor(return_2d_sequences(sequence)[0]).float().view(-1, 1)
            y_full = torch.tensor(return_2d_sequences(sequence)[1]).float().view(-1, 1)
            x_full_scaled = (x_full - x_full.mean()) / x_full.std()
            y_full_scaled = (y_full - y_full.mean()) / y_full.std()

            final_output = model(x_full_scaled)
            mse = criterion(final_output, y_full_scaled).item()
            mse_list.append(mse)

        # Store the average MSE for this percentage
        mean_mse = np.mean(mse_list)
        all_mse.append(mean_mse)
        print(f'Percentage: {percentage:.2f}, Average MSE: {mean_mse:.6f}')

    return percentages, all_mse

# Define your sequence and percentages
sequence = aquatic_bird_bornavirus_full  # Replace with actual sequence
percentages = [0.1, 0.15, 0.2,0.3,0.4,0.45, 0.5,0.6, 0.7,0.9, 1.0]  # Define your percentages here

# Call the function
percentages, mse_list = evaluate_sequence(sequence, percentages)

# Create a DataFrame to show the results
results_df = pd.DataFrame({
    'Percentage': percentages,
    'Average MSE': mse_list
})

# Print the table
print(results_df)

# Plot the Percentage vs. Average MSE
plt.figure(figsize=(10, 6))
plt.plot(percentages, mse_list, marker='o', linestyle='-', color='blue')
plt.xlabel('Percentage of sequence seen')
plt.ylabel('Average MSE')
plt.title('Percentage of sequence seen vs. Average MSE')
plt.grid(True)
#plt.show()
plt.savefig("outputs/perc-seen-seq-vs-context-win.png", format='png')

 # %%
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

input_size = 1
hidden_sizes = [32, 64, 128, 64, 32]
output_size = 1
learning_rate = 0.01
num_epochs = 1000
step_size = 200
gamma = 0.1

# Create the MLP model
model = MLP(input_size, hidden_sizes, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training data (example)
x_train = torch.tensor(return_2d_sequences(aquatic_bird_bornavirus_small)[0]).float().view(-1, 1)
y_train = torch.tensor(return_2d_sequences(aquatic_bird_bornavirus_small)[1]).float().view(-1, 1)

# Split the data into train and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Scale the input data to a smaller range
x_train_scaled = (x_train - x_train.mean()) / x_train.std()
y_train_scaled = (y_train - y_train.mean()) / y_train.std()

x_valid_scaled = (x_valid - x_train.mean()) / x_train.std()
y_valid_scaled = (y_valid - y_train.mean()) / y_train.std()

# Store the predicted values
y_pred_scaled = []

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train_scaled)
    loss = criterion(outputs, y_train_scaled)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the learning rate
    scheduler.step()

    # Evaluate on the validation set
    with torch.no_grad():
        valid_outputs = model(x_valid_scaled)
        valid_loss = criterion(valid_outputs, y_valid_scaled)

    # Store the predicted values
    y_pred_scaled.append(valid_outputs.numpy())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.6f}, Valid Loss: {valid_loss.item():.6f}, lr: {scheduler.get_last_lr()[0]:.6f}')

# Convert PyTorch tensors to NumPy arrays
y_valid_np = y_valid.numpy().flatten()
y_valid_std_np = y_valid.std().item()
y_valid_mean_np = y_valid.mean().item()

# Unscale the predicted values
y_pred = [y_pred_scaled[i] * y_valid_std_np + y_valid_mean_np for i in range(len(y_pred_scaled))]

# Plot the original and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(x_valid.numpy(), y_valid, label='Original')
plt.scatter(x_valid.numpy(), y_pred[-1], label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Original vs. Predicted Values (Validation Set)')
#plt.show()

# x_train_scaled

 # %%
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.hidden_layers.append(nn.GELU())
        self.hidden_layers.append(nn.Dropout(dropout_rate))  # Added dropout after GELU
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.hidden_layers.append(nn.GELU())
            self.hidden_layers.append(nn.Dropout(dropout_rate))  # Added dropout after GELU
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.output_layer(x)
        return out

# Normalize the labels (y) using StandardScaler
scaler = StandardScaler()
train_labels = scaler.fit_transform(train_labels.reshape(-1, 1))
test_labels = scaler.transform(test_labels.reshape(-1, 1))

# Convert data to PyTorch tensors
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Define the model and optimizer

input_size = train_inputs.shape[1]
hidden_sizes = [128, 256, 512]  # Example hidden layer sizes
output_size = 1

model = MLP(input_size, hidden_sizes, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)  # Added L2 regularization


# Train the model
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(train_inputs), batch_size):
        batch_inputs = train_inputs[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]

        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the test set
with torch.no_grad():
    test_outputs = model(test_inputs)
    test_outputs = scaler.inverse_transform(test_outputs.numpy())
    test_loss = criterion(torch.tensor(test_outputs), test_labels)
    print(f"Test Loss: {test_loss.item():.4f}")

# Generate predictions for a range of input values
input_range = np.linspace(train_inputs.min().item(), train_inputs.max().item(), 100)
input_range = torch.tensor(input_range, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    predicted_range = model(input_range)
    predicted_range = scaler.inverse_transform(predicted_range.numpy())

# Plot the original data and the learned function
plt.figure(figsize=(8, 6))
plt.scatter(train_inputs.numpy(), scaler.inverse_transform(train_labels.numpy()), label='Training Data')
plt.scatter(test_inputs.numpy(), scaler.inverse_transform(test_labels.numpy()), label='Test Data')
plt.plot(input_range.numpy(), predicted_range, color='red', label='Learned Function')

# Plot the original data
original_inputs = np.concatenate((train_inputs.numpy(), test_inputs.numpy()))
original_labels = np.concatenate((scaler.inverse_transform(train_labels.numpy()), scaler.inverse_transform(test_labels.numpy())))
#plt.scatter(original_inputs, original_labels, color='green', label='Original Data')

plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Learned Function and Original Data')
plt.legend()
#plt.show()
plt.savefig("outputs/aquatic_bird_bornavirus_full.png", format='png')


 # %%
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Normalize the labels (y) using StandardScaler
scaler = StandardScaler()
train_labels = scaler.fit_transform(train_labels.reshape(-1, 1))
test_labels = scaler.transform(test_labels.reshape(-1, 1))

# Convert data to PyTorch tensors
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Define the model and optimizer
input_size = train_inputs.shape[1]
hidden_size = 64
output_size = 1

model = MLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(train_inputs), batch_size):
        batch_inputs = train_inputs[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]

        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model on the test set
with torch.no_grad():
    test_outputs = model(test_inputs)
    test_outputs = scaler.inverse_transform(test_outputs.numpy())
    test_loss = criterion(torch.tensor(test_outputs), test_labels)
    print(f"Test Loss: {test_loss.item():.4f}")

# Generate predictions for a range of input values
input_range = np.linspace(train_inputs.min().item(), train_inputs.max().item(), 100)
input_range = torch.tensor(input_range, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    predicted_range = model(input_range)
    predicted_range = scaler.inverse_transform(predicted_range.numpy())

# Plot the original data and the learned function
plt.figure(figsize=(8, 6))
plt.scatter(train_inputs.numpy(), scaler.inverse_transform(train_labels.numpy()), label='Training Data', alpha=0.5, s=6)
plt.scatter(test_inputs.numpy(), scaler.inverse_transform(test_labels.numpy()), label='Test Data', alpha=0.5, s=6)
plt.plot(input_range.numpy(), predicted_range, color='red', label='Learned Function')

# Plot the original data
original_inputs = np.concatenate((train_inputs.numpy(), test_inputs.numpy()))
original_labels = np.concatenate((scaler.inverse_transform(train_labels.numpy()), scaler.inverse_transform(test_labels.numpy())))
# plt.scatter(original_inputs, original_labels, color='green', label='Original Data', alpha=0.5, marker="+")

plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Learned Function and Original Data')
plt.legend()
#plt.show()
plt.savefig("outputs/MLP_learned_function.png", format='png')


# %%

from kan import KAN, create_dataset
import torch
import seaborn as sns

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[1,3,3,3,1], grid=10, k=5, seed=0)
f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
dataset = create_dataset(f, n_var=4, train_num=3000)
dataset['test_label']=y_train_scaled
dataset['train_label']=y_train_scaled
dataset['test_input']=x_train_scaled
dataset['train_input']=x_train_scaled

# train the model
model.train(dataset, opt="LBFGS", steps=200, lamb=0.001, lamb_entropy=2.);

model.plot()

model.prune()

model = model.prune()
model(dataset['train_input'])
model.plot()

model.suggest_symbolic(1,0,0,topk=15)

model.auto_symbolic()

model.train(dataset, opt="LBFGS", steps=20);
model.plot()

model.auto_symbolic()


sns.lineplot(x= dataset['train_input'].flatten(), y = dataset['train_label'].flatten())


lowpassfilter(return_2d_sequences(aquatic_bird_bornavirus_full)[1])

return_2d_sequences(aquatic_bird_bornavirus_full)[1]

model.plot()

dataset['test_label']=test_labels
dataset['train_label']=train_labels
dataset['test_input']=test_inputs
dataset['train_input']=train_inputs
