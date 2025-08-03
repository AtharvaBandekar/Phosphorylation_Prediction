# Import required modules
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt

# Import model and mappings required
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from models.cnn_predictor import PhosphoPredictorCNN

from scripts.data_preprocessing import num_amino_acids, aa_to_int 
amino_acids_list = list(aa_to_int.keys())

motif_save_dir = os.path.join(project_root, 'results', 'motifs')
os.makedirs(motif_save_dir, exist_ok=True)

# Configurations
SEQUENCE_LENGTH = 21
KERNEL_SIZE = 5 
NUM_AMINO_ACIDS = num_amino_acids

# Set up the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS (GPU) for computations.")
else:
    device = torch.device("cpu")
    print("Using CPU for computations.")

# Load the trained model
model_save_path = os.path.join(project_root, 'models', 'best_phospho_predictor.pth')
model = PhosphoPredictorCNN(num_amino_acids=NUM_AMINO_ACIDS, sequence_length=SEQUENCE_LENGTH, kernel_size=KERNEL_SIZE).to(device)

try:
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"Loaded best model from {model_save_path}")
except FileNotFoundError:
    print(f"Error: Best model not found at {model_save_path}. Please ensure training completed successfully.")
    exit()

model.eval()

# Extract weights from the fist CNN layers
first_conv_weights = model.conv1.weight.data.cpu().numpy()
print(f"Shape of first convolutional layer weights: {first_conv_weights.shape}") 

# Motif Vizualization
num_filters_to_visualize = 10 

for i in range(min(num_filters_to_visualize, first_conv_weights.shape[0])):
    filter_weights = first_conv_weights[i, :, :] 

    information_matrix = pd.DataFrame(filter_weights.T, columns=amino_acids_list)

    plt.figure(figsize=(10, 3))
    logo = logomaker.Logo(information_matrix, font_name='Arial Rounded MT Bold', vpad=0.1, width=0.8)
    
    logo.style_spines(visible=False)
    logo.ax.set_xticks(range(KERNEL_SIZE))
    logo.ax.set_xticklabels([str(j) for j in range(KERNEL_SIZE)])
    logo.ax.set_ylabel("Weight")
    logo.ax.yaxis.label.set_color('black')
    logo.ax.set_title(f"Learned Motif for Filter {i+1} (Weights from Conv1)", fontsize=14)
    plot_filename = os.path.join(motif_save_dir, f"filter_{i+1}_motif.png")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close(plt.gcf())


print("Motif visualization complete.")
