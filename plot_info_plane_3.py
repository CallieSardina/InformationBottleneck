import re
import numpy as np
import matplotlib.pyplot as plt

# Initialize the variables
epochs = []
mi_xz_1 = []
mi_xz_2 = []
mi_xz_3 = []
mi_zy_1 = []
mi_zy_2 = []
mi_zy_3 = []

# Read the data from file
file_path = 'run_results.txt'  # Replace with your actual file path
with open(file_path, 'r') as file:
    for line in file:
        # Match the pattern in each line using regex
        match = re.match(r'Epoch (\d+), Batch (\d+), MI_XZ: ([\d\.]+), ([\d\.]+), ([\d\.]+), MI_ZY: ([\d\.]+), ([\d\.]+), ([\d\.]+)', line)
        if match:
            # Extract the values and append to lists
            epoch = int(match.group(1))
            batch = int(match.group(2))
            mi_xz_values = [float(match.group(3)), float(match.group(4)), float(match.group(5))]
            mi_zy_values = [float(match.group(6)), float(match.group(7)), float(match.group(8))]  

            epochs.append(epoch)
            mi_xz_1.append(mi_xz_values[0])  # For each layer's MI_XZ value
            mi_xz_2.append(mi_xz_values[1]) 
            mi_xz_3.append(mi_xz_values[2]) 
            mi_zy_1.append(mi_zy_values[0])  
            mi_zy_2.append(mi_zy_values[1])  
            mi_zy_3.append(mi_zy_values[2])  

# Convert to numpy arrays for easier manipulation
epochs = np.array(epochs)
mi_xz_1 = np.array(mi_xz_1)
mi_xz_2 = np.array(mi_xz_2)
mi_xz_3 = np.array(mi_xz_3)
mi_zy_1 = np.array(mi_zy_1)
mi_zy_2 = np.array(mi_zy_2)
mi_zy_3 = np.array(mi_zy_3)

# Set maximum epoch for colorbar range
COLORBAR_MAX_EPOCHS = max(epochs)

# Create the colormap scalar mappable
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
sm._A = []

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Loop over each epoch
for epoch in sorted(set(epochs)):
    # Get the color corresponding to the epoch
    c = sm.to_rgba(epoch)

    # Calculate the average MI_XZ and MI_ZY values for the current epoch for each layer
    avg_mi_xz_1 = np.mean(mi_xz_1[epochs == epoch])
    avg_mi_xz_2 = np.mean(mi_xz_2[epochs == epoch])
    avg_mi_xz_3 = np.mean(mi_xz_3[epochs == epoch])
    
    avg_mi_zy_1 = np.mean(mi_zy_1[epochs == epoch])
    avg_mi_zy_2 = np.mean(mi_zy_2[epochs == epoch])
    avg_mi_zy_3 = np.mean(mi_zy_3[epochs == epoch])

    # Plot the average data point for each layer
    # ax.plot([avg_mi_xz_1, avg_mi_xz_2, avg_mi_xz_3], [avg_mi_zy_1, avg_mi_zy_2, avg_mi_zy_3], 
    #         c=c, alpha=0.7, marker='o', label=f'Epoch {epoch}' if epoch == sorted(set(epochs))[0] else "")
    ax.plot([avg_mi_xz_1, avg_mi_xz_2, avg_mi_xz_3], 
            [avg_mi_zy_1, avg_mi_zy_2, avg_mi_zy_3], 
            c=c, alpha=0.1, zorder=1, label=f'Epoch {epoch}' if epoch == sorted(set(epochs))[0] else "")
    
    # Scatter plot for the points (less opaque)
    ax.scatter([avg_mi_xz_1, avg_mi_xz_2, avg_mi_xz_3], 
               [avg_mi_zy_1, avg_mi_zy_2, avg_mi_zy_3], 
               s=30, facecolors=[c]*3, edgecolor='none', alpha=1, zorder=2)  # Lower alpha for points

# Set labels and title
ax.set_xlabel('I(X; Z)')
ax.set_ylabel('I(Z; Y)')
ax.set_title('Info Plane Across Layers')

# Add a colorbar for epoch values
cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
plt.colorbar(sm, label='Epoch', cax=cbaxes)

# Adjust layout and show/save the plot
plt.tight_layout()

# If you want to save the plot
DO_SAVE = False  # Set this to True if you want to save the plot
if DO_SAVE:
    plt.savefig('plots/infoplane_layers_all_on_same_plot.png', bbox_inches='tight')

plt.show()
