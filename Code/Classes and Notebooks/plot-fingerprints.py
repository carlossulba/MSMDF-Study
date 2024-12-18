import os
import random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Machine Learning Imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from FingerprintExtractor import FingerprintConfig, FingerprintSetting, FingerprintSensor, FingerprintFeature, FingerprintDataStream


class FingerprintVisualizer:
    def __init__(self, data_path, random_seed=42):
        """ Initialize the visualizer with data and random seed. """
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Load data
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        self.data = data['fingerprints']
        
        # Prepare output directory
        self.output_dir = self._prepare_output_directory()

    def _prepare_output_directory(self):
        """ Create a timestamped output directory for plots. """
        output_dir = "Results/FingerprintExtractor/Plots/"
        current_date = datetime.now().strftime('%Y-%m-%d__%H-%M')
        dated_save_directory = os.path.join(output_dir, current_date)
        os.makedirs(dated_save_directory, exist_ok=True)
        return dated_save_directory

    def _prepare_fingerprints(self):
        """ Prepare fingerprints, labels, and device information. """
        fingerprints = []
        labels = []
        device_list = []
        device_labels = []

        device_counter = 1
        for setting, devices in self.data.items():
            for device, fingerprints_device in devices.items():
                for fingerprint in fingerprints_device:
                    fingerprint_list = fingerprint[0]
                    for single_fingerprint in fingerprint_list:
                        # Convert to numpy array and set NaNs to 0
                        single_fingerprint = np.nan_to_num(np.array(single_fingerprint), nan=0.0)
                        fingerprints.append(single_fingerprint.flatten())
                        labels.append(f"{setting}_{device}")
                        device_list.append(device)
                        device_labels.append(f"{device_counter}")
                device_counter += 1

        # Create DataFrame and shuffle
        df = pd.DataFrame({
            'fingerprints': list(np.array(fingerprints)),
            'labels': labels,
            'device_list': device_list,
            'device_labels': device_labels
        })
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return (
            np.array(df['fingerprints'].tolist()), 
            df['labels'].tolist(), 
            df['device_list'].tolist(), 
            df['device_labels'].tolist()
        )

    def _plot_dimensionality_reduction(self, reduction_result, reduction_type, device_list, device_labels, save=False):
        """ Create and save a plot for dimensionality reduction results. """
        # Create color map
        unique_devices = list(set(device_list))
        cmap = plt.colormaps["hsv"]
        device_to_color = {device: cmap(i / len(unique_devices)) for i, device in enumerate(unique_devices)}
        
        device_colors = [device_to_color[device] for device in device_list]
        unique_devices = list(set(device_list))

        # Create plot
        plt.figure(figsize=(12, 8))
        plt.scatter(reduction_result[:, 0], reduction_result[:, 1], c=device_colors, alpha=0.7)
        
        # Track the count of labels per device
        device_label_count = {}
        max_labels_per_device = 4

        # Add device number labels to half the points, with a max of 5 per device
        for i, (x, y) in enumerate(zip(reduction_result[:, 0], reduction_result[:, 1])):
            device = device_labels[i]
            
            # Initialize the counter for the device if not already present
            if device not in device_label_count:
                device_label_count[device] = 0
            
            # Add label only if it's one of the half points and hasn't exceeded the max limit
            if i % 2 == 0 and device_label_count[device] < max_labels_per_device:
                plt.text(x, y, f"{device}", fontsize=3, ha='center', color=(0.3, 0.3, 0.3), zorder=5)
                device_label_count[device] += 1

        
        plt.title(f'2D {reduction_type} of High Dimensional Fingerprints')
        plt.xlabel(f'{reduction_type} Component 1')
        plt.ylabel(f'{reduction_type} Component 2')
        
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.output_dir, f"{reduction_type.lower()}_fingerprints.png")
        if save:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def visualize(self, save: bool = False):
        """ Perform dimensionality reduction and visualization. """
        # Prepare fingerprints
        fingerprints, labels, device_list, device_labels = self._prepare_fingerprints()

        # Standardize the data
        scaler = StandardScaler()
        fingerprints_scaled = scaler.fit_transform(fingerprints)

        # PCA Visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(fingerprints_scaled)
        self._plot_dimensionality_reduction(pca_result, 'PCA', device_list, device_labels, save)

        # t-SNE Visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(fingerprints_scaled)
        self._plot_dimensionality_reduction(tsne_result, 't-SNE', device_list, device_labels, save)
        
        if save:
            print("Plots saved successfully!")

def main():
    # Path to your fingerprint data pickle file
    data_path = "../Data/extracted fingerprints/extracted_fingerprints_2024-12-18__13-57.pkl"
    
    # Create visualizer and generate plots
    visualizer = FingerprintVisualizer(data_path)
    visualizer.visualize(save=True)

if __name__ == "__main__":
    main()