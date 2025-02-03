import re
import matplotlib.pyplot as plt
import numpy as np

# File path where data is stored
log_file = 'run_results_with_fc_layers.txt'

# Initialize lists to store MI_XZ, MI_ZY, and Z_i (layer) values
mi_xz_values = []
mi_zy_values = []
zi_values = []
epochs = []

# Read the log file and extract values
with open(log_file, 'r') as file:
    for line in file:
        # Use regex to extract the relevant values from each line
        match = re.match(r"Epoch (\d+), Batch (\d+), MI_XZ:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+), MI_ZY:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)", line)
        if match:
            # Extract the values
            epoch = int(match.group(1))
            batch = int(match.group(2))
            mi_xz1 = float(match.group(3))
            mi_xz2 = float(match.group(4))
            mi_xz3 = float(match.group(5))
            mi_xz4 = float(match.group(6))
            mi_xz5 = float(match.group(7))


            mi_zy1 = float(match.group(8))
            mi_zy2 = float(match.group(9))
            mi_zy3 = float(match.group(10))
            mi_zy4 = float(match.group(11))
            mi_zy5 = float(match.group(12))

            # Append the extracted values to the lists
            mi_xz_values.append((mi_xz1, mi_xz2, mi_xz3, mi_xz4, mi_xz5))
            mi_zy_values.append((mi_zy1, mi_zy2, mi_zy3, mi_zy4, mi_zy5))
            zi_values.append([1, 2, 3, 4, 5])  # Z_i is just the layer number (1, 2, 3) for each entry
            epochs.append(epoch)

# Convert the lists to numpy arrays for easier manipulation
mi_xz_values = np.array(mi_xz_values)
mi_zy_values = np.array(mi_zy_values)
zi_values = np.array(zi_values)
epochs = np.array(epochs)

# Create subplots: one for MI_XZ and one for MI_ZY
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Two subplots

COLORBAR_MAX_EPOCHS = max(epochs)
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
sm._A = []

# Plotting MI_XZ vs Z_i for each epoch and layer (in the first subplot)
for epoch in sorted(set(epochs)):  # Iterate through epochs
    epoch_mask = epochs == epoch
    # Extract the values for the current epoch
    mi_xz_epoch = mi_xz_values[epoch_mask][0]  # We are assuming only one entry per epoch
    mi_zy_epoch = mi_zy_values[epoch_mask][0]  # We are assuming only one entry per epoch
    layers = zi_values[epoch_mask][0]  # Layers are always [1, 2, 3]

    c = sm.to_rgba(epoch)

    # Scatter plot for the layers and MI_XZ values
    axes[0].scatter(layers, mi_xz_epoch, label=f'Epoch {epoch}', alpha=0.6, facecolors=[c])
    
    # Connect points across layers (1 → 2 → 3) for the same epoch
    axes[0].plot(layers, mi_xz_epoch, linestyle='-', color=c, alpha=0.4)

axes[0].set_xlabel('Layer (Z_i)')
axes[0].set_ylabel('MI_XZ Values')
axes[0].set_title('MI_XZ as a function of Z_i')

# Plotting MI_ZY vs Z_i for each epoch and layer (in the second subplot)
for epoch in sorted(set(epochs)):  # Iterate through epochs
    epoch_mask = epochs == epoch
    # Extract the values for the current epoch
    mi_xz_epoch = mi_xz_values[epoch_mask][0]  # We are assuming only one entry per epoch
    mi_zy_epoch = mi_zy_values[epoch_mask][0]  # We are assuming only one entry per epoch
    layers = zi_values[epoch_mask][0]  # Layers are always [1, 2, 3]

    c = sm.to_rgba(epoch)

    # Scatter plot for the layers and MI_ZY values
    axes[1].scatter(layers, mi_zy_epoch, label=f'Epoch {epoch}', alpha=0.6, facecolors=[c])
    
    # Connect points across layers (1 → 2 → 3) for the same epoch
    axes[1].plot(layers, mi_zy_epoch, linestyle='-', color=c, alpha=0.4)

axes[1].set_xlabel('Layer (Z_i)')
axes[1].set_ylabel('MI_ZY Values')
axes[1].set_title('MI_ZY as a function of Z_i')

cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
plt.colorbar(sm, label='Epoch', cax=cbaxes)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# import re
# import matplotlib.pyplot as plt

# # Path to the file
# file_path = 'run_results.txt'  # Replace with your actual file path

# # Initialize variables to store MI_XZ and MI_ZY values for each layer
# MI_XZ_values = {1: [], 2: [], 3: []}
# MI_ZY_values = {1: [], 2: [], 3: []}

# # Read the data from the file and extract MI_XZ and MI_ZY values
# with open(file_path, 'r') as file:
#     for line in file:
#         # Match the pattern in each line using regex
#         match = re.match(r'Epoch (\d+), Batch (\d+), MI_XZ: ([\d\.]+), ([\d\.]+), ([\d\.]+), MI_ZY: ([\d\.]+), ([\d\.]+), ([\d\.]+)', line)
        
#         if match:
#             epoch = int(match.group(1))
#             batch = int(match.group(2))
            
#             # MI_XZ values for layers 1, 2, 3
#             mi_xz_1 = float(match.group(3))
#             mi_xz_2 = float(match.group(4))
#             mi_xz_3 = float(match.group(5))
            
#             # MI_ZY values for layers 1, 2, 3
#             mi_zy_1 = float(match.group(6))
#             mi_zy_2 = float(match.group(7))
#             mi_zy_3 = float(match.group(8))
            
#             # Append the values to the corresponding layer's lists
#             MI_XZ_values[1].append(mi_xz_1)
#             MI_XZ_values[2].append(mi_xz_2)
#             MI_XZ_values[3].append(mi_xz_3)
            
#             MI_ZY_values[1].append(mi_zy_1)
#             MI_ZY_values[2].append(mi_zy_2)
#             MI_ZY_values[3].append(mi_zy_3)

# # Create a single plot showing MI_ZY vs MI_XZ for all layers
# plt.figure(figsize=(8, 6))

# # Plot each layer with a different marker or color
# plt.scatter(MI_XZ_values[1], MI_ZY_values[1], label='Layer 1', alpha=0.7, color='r', marker='o')
# plt.scatter(MI_XZ_values[2], MI_ZY_values[2], label='Layer 2', alpha=0.7, color='g', marker='x')
# plt.scatter(MI_XZ_values[3], MI_ZY_values[3], label='Layer 3', alpha=0.7, color='b', marker='s')

# # Set labels and title
# plt.xlabel('MI_XZ')
# plt.ylabel('MI_ZY')
# plt.title('MI_ZY vs MI_XZ for Layers 1, 2, and 3')

# # Add legend to distinguish layers
# plt.legend()

# # Adjust layout
# plt.tight_layout()

# # Optionally, save the plot
# DO_SAVE = False
# if DO_SAVE:
#     plt.savefig('plots/mi_z_y_vs_mi_x_z.png', bbox_inches='tight')

# # Show the plot
# plt.show()
