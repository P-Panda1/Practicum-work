import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from cftsdata import abr 
from pathlib import Path
import os
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.exploratory.visualization import FPCAPlot


mouse_id = 114
time_point = 1
ear = 'right'
frequency = 4000
# mouse_id = 89
# time_point = 0
# ear = 'right'
# frequency = 8000

# path to the data file
filename = Path(f'../../../../data/practicum-data/ABRpresto data/ABRpresto data/Mouse{mouse_id}_timepoint{time_point}_{ear} abr_io').absolute()
print(f'Loading {filename}')

fh = abr.load(filename)
epochs = fh.get_epochs_filtered()

# path to thresholds
thresholds = pd.read_csv('~/data/practicum-data/ABRpresto data/ABRpresto data/Manual Thresholds.csv')

thresholds.head()

epochs.head()

def extract_data(mouse_id=114, time_point=0, ear='right'):
    """
    Extracts the epochs and thresholds for a given mouse, timepoint and ear.
    """
    # path to the data file
    filename = Path(f'../../../../data/practicum-data/ABRpresto data/ABRpresto data/Mouse{mouse_id}_timepoint{time_point}_{ear} abr_io').absolute()
    print(f'Loading {filename}')

    fh = abr.load(filename)
    ep = fh.get_epochs_filtered()

    # extract thresholds and frequency pair for same mouse, timepoint and ear 
    th = thresholds[(thresholds['id'] == mouse_id) & (thresholds['timepoint'] == time_point) & (thresholds['ear'] == ear)][['frequency', 'manual threshold']]
    # th = thresholds[(thresholds['id'] == mouse_id) & (thresholds['timepoint'] == time_point) & (thresholds['ear'] == ear)]['frequency', 'manual threshold']
    return ep, th

abr_data, abr_thresholds = extract_data(mouse_id, time_point, ear)

abr_data.head()

abr_thresholds.head()

abr_data_group = abr_data.groupby(['frequency', 'level'])
abr_data_group.head()
abr_list = []
for (freq, level), group in abr_data_group:
    if freq == 16000:  # Your target frequency
        # abr_list = abr_list.append({
        #     'wave': group.values,  # Replace with actual waveform column
        #     'level': level
        # }, ignore_index=True)
        #Append the group and level to the list
        abr_list.append((group.values, level))
        
print(abr_list[0][0].shape)

def abr_list_for_freq(abr_data, abr_thresholds, frequency=4000):
    """
    Extracts the ABR data for a given
    frequency from the abr_data and abr_thresholds
    """
    abr_data_group = abr_data.groupby(['frequency', 'level'])
    abr_list = []
    th = abr_thresholds[abr_thresholds['frequency'] == frequency]['manual threshold'].values[0]
    for (freq, level), group in abr_data_group:
        if freq == frequency:  # Your target frequency
            #Append the group and level to the list
            abr_list.append((group.values, level))
    return abr_list, th

def plot_abr_for_freq(abr_data, abr_thresholds, frequency=4000):
    """
    Plots the ABR data for a given frequency
    """
    abr_list, th = abr_list_for_freq(abr_data, abr_thresholds, frequency)
    print(f'Threshold for {frequency} Hz is {th} dB')
    #Plot the ABR data
    for abr_wave, level in abr_list:
        plt.figure(figsize=(12, 10))
        mean = np.mean(abr_wave, axis=0)
        for i, row in enumerate(abr_wave):
            # Plot the waveforms for one level on top of each other
            plt.plot(row)
            plt.title(f'ABR waveforms for {level} dB')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
        # Plot the mean waveform
        plt.plot(mean, label='Mean', color='black')
        plt.show()

plot_abr_for_freq(abr_data, abr_thresholds, frequency)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction.projection import FPCA
import scipy.integrate

def apply_fpca_per_level(abr_data, abr_thresholds, frequency=4000, n_components=3, selected_level=None):
    """
    Applies Functional Principal Component Analysis (FPCA) on the final 3 levels of ABR waveforms for a given frequency.
    Prints the percentage of reconstruction variance retained for each component.
    """
    # Extract ABR waveforms for the given frequency
    abr_list, th = abr_list_for_freq(abr_data, abr_thresholds, frequency)

    # Only process the last 3 levels
    abr_list = abr_list[-3:]

    # Initialize dictionary to hold FPCA results for each level
    fpca_results = {}

    # Process each level

    for abr_wave, level in abr_list:
        # If level is None print all levels, if specified only process that level
        if selected_level is not None and selected_level != level:
            continue

        print(f"\nProcessing ABR waveforms for {frequency} Hz at {level} dB...\n")

        # Convert list of waveforms into an array for this level
        all_waveforms = np.array(abr_wave)

        # Ensure the array is in the correct format
        if all_waveforms.ndim == 1:
            print('Reshaping array')
            all_waveforms = all_waveforms.reshape(-1, 1)

        # Create FDataGrid for FPCA
        time_points = np.linspace(0, 1, all_waveforms.shape[1])  # Normalize time points
        fdata = FDataGrid(data_matrix=all_waveforms, grid_points=[time_points])

        # Compute trapezoidal weights
        dx = np.diff(time_points)
        weights = np.zeros_like(time_points)
        weights[0] = dx[0] / 2
        weights[-1] = dx[-1] / 2
        weights[1:-1] = (dx[:-1] + dx[1:]) / 2  # Middle weights

        # Apply FPCA with explicit weights
        fpca = FPCA(n_components=n_components)
        fpca._weights = weights  # Manually assign weights
        fpca.fit(fdata)
        transformed_data = fpca.transform(fdata)

        # Store FPCA results for this level
        fpca_results[level] = {
            'transformed_data': transformed_data,
            'fpca': fpca
        }

        # Print reconstruction percentage for each component count
        explained_var = fpca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        for k, var in enumerate(cumulative_var, start=1):
            print(f"Using {k} FPCA components: {var * 100:.2f}% of signal variance retained.")

        # Scatter plot of first two FPCA components
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=[level] * len(transformed_data), cmap='viridis')
        ax.set_xlabel('FPCA Component 1')
        ax.set_ylabel('FPCA Component 2')
        ax.set_title(f'FPCA of ABR waveforms for {frequency} Hz at {level} dB')
        plt.colorbar(scatter, label="dB Level")
        plt.show()

        # Scree plot (explained variance)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linestyle='-')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained')
        ax.set_title(f'Scree Plot for {frequency} Hz at {level} dB')
        ax.grid(True)
        plt.show()

    return fpca_results


# Apply FPCA for each level
fpca_results = apply_fpca_per_level(abr_data, abr_thresholds, frequency, 20, 105)

def plot_fpca_analysis(level, abr_list, fpca_results, eigenvecNo=3):
    """
    Plots FPCA analysis for a specific level including eigenvectors, component distribution,
    and extreme waveform examples with reconstructions.
    """
    # Get FPCA results and original data for the specified level
    level_results = fpca_results[level]
    fpca = level_results['fpca']
    transformed_data = level_results['transformed_data']
    
    # Get original waveforms for this level
    abr_wave = next(wave for wave, lvl in abr_list if lvl == level)
    waveforms = np.array(abr_wave)
    time_points = np.linspace(0, 1, waveforms.shape[1])
    
    # 1. Plot eigenvectors (principal components)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i, component in enumerate(fpca.components_):
        if i == eigenvecNo:
            break
        ax1.plot(time_points, component.data_matrix[0, :, 0], 
                label=f'Component {i+1}')
    ax1.set_title(f'FPCA Components - {level} dB')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    plt.show()

    # 2. Plot distribution with extremes numbered
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    scatter = ax2.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.6)
    
    # Find extreme points in all directions
    extremes = {
        'top': np.argmax(transformed_data[:, 1]),
        'bottom': np.argmin(transformed_data[:, 1]),
        'right': np.argmax(transformed_data[:, 0]),
        'left': np.argmin(transformed_data[:, 0])
    }
    
    # Annotate extreme points
    for direction, idx in extremes.items():
        ax2.annotate(str(idx), 
                    (transformed_data[idx, 0], transformed_data[idx, 1]),
                    textcoords="offset points",
                    xytext=(0,10 if direction == 'top' else -15),
                    ha='center')
    
    ax2.set_title(f'FPCA Component Distribution - {level} dB\nExtreme Points Numbered')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    plt.show()

    # 3. Plot extreme waveforms with reconstructions
    fig3, axs = plt.subplots(2, 2, figsize=(15, 10))
    extreme_indices = list(extremes.values())
    
    for i, idx in enumerate(extreme_indices[:4]):  # Plot first 4 extremes
        ax = axs[i//2, i%2]
        
        # Original and reconstructed waveforms
        original = waveforms[idx]
        reconstructed = fpca.inverse_transform(transformed_data[idx:idx+1])[0].data_matrix[0, :, 0]
        
        ax.plot(time_points, original, label='Original')
        ax.plot(time_points, reconstructed, '--', label='Reconstructed')
        ax.set_title(f'Extreme {idx} - {list(extremes.keys())[i]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

plot_fpca_analysis(100.0, abr_list, fpca_results, eigenvecNo=4)

plot_fpca_analysis(105.0, abr_list, fpca_results)

from sklearn.cluster import KMeans, DBSCAN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

def cluster_and_plot_kmeans(level, abr_list, fpca_results, n_groups=3, **kwargs):
    """
    Clusters FPCA results using K-means with separate plots for distribution and waveforms.
    Returns a dictionary mapping cluster IDs to waveform indices.
    """
    # Get FPCA data for specified level
    level_data = fpca_results[level]
    fpca = level_data['fpca']
    transformed_data = level_data['transformed_data']
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_groups, **kwargs)
    labels = kmeans.fit_predict(transformed_data)
    
    # Get original waveforms
    abr_wave = next(wave for wave, lvl in abr_list if lvl == level)
    waveforms = np.array(abr_wave)
    time_points = np.linspace(0, 1, waveforms.shape[1])
    
    # Store clusters in dictionary
    cluster_dict = {i: np.where(labels == i)[0].tolist() for i in range(n_groups)}

    # Plot cluster distribution
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    scatter = ax1.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis')
    ax1.set_title(f'K-means Clustering (n={n_groups}) - {level} dB')
    ax1.set_xlabel('FPCA Component 1')
    ax1.set_ylabel('FPCA Component 2')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    plt.show()

    # Create waveform plot
    fig2 = plt.figure(figsize=(15, 4 * n_groups))
    cmap = scatter.cmap
    norm = scatter.norm
    
    for cluster_id in range(n_groups):
        cluster_mask = (labels == cluster_id)
        if not cluster_mask.any():
            continue
        
        # Find representative waveform (closest to cluster center)
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(transformed_data[cluster_mask] - centroid, axis=1)
        rep_idx = np.argmin(distances)
        original_idx = np.where(cluster_mask)[0][rep_idx]

        ax = fig2.add_subplot(n_groups, 1, cluster_id + 1)
        color = cmap(norm(cluster_id))
        original = waveforms[original_idx]
        reconstructed = fpca.inverse_transform(transformed_data[[original_idx]]).data_matrix[0, :, 0]
        
        ax.plot(time_points, original, color=color, label='Original')
        ax.plot(time_points, reconstructed, '--', color=color, label='Reconstructed')
        ax.set_title(f'Cluster {cluster_id} Representative (Index {original_idx})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    plt.show()

    return {cluster_id: np.where(labels == cluster_id)[0].tolist() for cluster_id in set(labels)}

def cluster_and_plot_dbscan(level, abr_list, fpca_results, eps=0.5, min_samples=5, **kwargs):
    """
    Clusters FPCA results using DBSCAN with separate plots for distribution and waveforms.
    Returns a dictionary mapping cluster IDs to waveform indices.
    """
    # Get FPCA data for specified level
    level_data = fpca_results[level]
    fpca = level_data['fpca']
    transformed_data = level_data['transformed_data']
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = dbscan.fit_predict(transformed_data)

    # Get original waveforms
    abr_wave = next(wave for wave, lvl in abr_list if lvl == level)
    waveforms = np.array(abr_wave)
    time_points = np.linspace(0, 1, waveforms.shape[1])

    # Store clusters in dictionary
    cluster_dict = {i: np.where(labels == i)[0].tolist() for i in set(labels)}

    # Plot cluster distribution
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    scatter = ax1.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis', vmin=min(labels)-1, vmax=max(labels)+1)
    ax1.set_title(f'DBSCAN Clustering - {level} dB\n(eps={eps}, min_samples={min_samples})')
    ax1.set_xlabel('FPCA Component 1')
    ax1.set_ylabel('FPCA Component 2')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    plt.show()

    # Create waveform plot
    unique_labels = set(labels) - {-1}
    fig2 = plt.figure(figsize=(15, 4 * len(unique_labels)))
    cmap = scatter.cmap
    norm = scatter.norm

    for i, cluster_id in enumerate(unique_labels):
        cluster_mask = (labels == cluster_id)
        
        if not cluster_mask.any():
            continue
        
        # Find medoid (most central point)
        pairwise_dist = np.linalg.norm(transformed_data[cluster_mask][:, None] - transformed_data[cluster_mask], axis=2)
        medoid_idx = np.argmin(pairwise_dist.sum(axis=0))
        original_idx = np.where(cluster_mask)[0][medoid_idx]

        ax = fig2.add_subplot(len(unique_labels), 1, i + 1)
        color = cmap(norm(cluster_id))
        original = waveforms[original_idx]
        reconstructed = fpca.inverse_transform(transformed_data[[original_idx]]).data_matrix[0, :, 0]
        
        ax.plot(time_points, original, color=color, label='Original')
        ax.plot(time_points, reconstructed, '--', color=color, label='Reconstructed')
        ax.set_title(f'Cluster {cluster_id} Medoid (Index {original_idx})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    plt.show()

    return {cluster_id: np.where(labels == cluster_id)[0].tolist() for cluster_id in set(labels)}


kmeans_dict = cluster_and_plot_kmeans(105.0, abr_list, fpca_results, n_groups=3)

dbscan_dict = cluster_and_plot_dbscan(105.0, abr_list, fpca_results, eps=0.5*(1e-5), min_samples=5)

import seaborn as sns

def apply_fpca_on_clusters(cluster_dict, abr_list, level, n_components=5):
    """
    Applies Functional Principal Component Analysis (FPCA) on each cluster's subset of waveforms.
    Plots scree plots and component score distributions.

    Parameters:
    - cluster_dict: Dictionary mapping cluster IDs to waveform indices
    - abr_list: List of (waveforms, level) tuples
    - level: dB level to process
    - n_components: Number of functional principal components to retain
    """
    # Get original waveforms for the given level
    abr_wave = next(wave for wave, lvl in abr_list if lvl == level)
    waveforms = np.array(abr_wave)

    # Ensure valid shape
    if waveforms.ndim == 1:
        waveforms = waveforms.reshape(-1, 1)

    # Time points for functional data representation
    time_points = np.linspace(0, 1, waveforms.shape[1])

    # Dictionary to store FPCA results
    fpca_results = {}

    for cluster_id, indices in cluster_dict.items():
        if cluster_id == -1:
            print(f"Skipping noise cluster (-1) for level {level} dB.")
            continue

        if len(indices) < n_components:
            print(f"Skipping FPCA on Cluster {cluster_id} because the number of samples {len(indices)} is less than {n_components}")
            continue

        print(f"\nPerforming FPCA on Cluster {cluster_id} (Level {level} dB) with {len(indices)} samples...")

        # Extract subset of waveforms for this cluster
        cluster_waveforms = waveforms[indices]

        # Convert to functional data format
        fdata = FDataGrid(data_matrix=cluster_waveforms, grid_points=[time_points])

        # Apply FPCA
        fpca = FPCA(n_components=n_components)
        transformed_data = fpca.fit_transform(fdata)

        # Ensure explained variance sums to 1
        explained_variance = fpca.explained_variance_ratio_

        # Compute cumulative variance
        cumulative_var = np.cumsum(explained_variance)

        # Store results
        fpca_results[cluster_id] = {
            'transformed_data': transformed_data,
            'fpca': fpca,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_var
        }

        # Print explained variance per component
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_var)):
            print(f"  Component {i+1}: {var * 100:.4f} variance explained (Cumulative: {cum_var * 100:.4f})")

        # Plot scree plot
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-', label="Explained Variance")
        # plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='s', linestyle='--', label="Cumulative Variance")
        plt.xlabel('Functional Principal Component')
        plt.ylabel('Variance Explained')
        plt.title(f'Scree Plot - Cluster {cluster_id} (Level {level} dB)')
        plt.legend()
        plt.grid()
        plt.show()

         # Scatter plot of first two FPCA components
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=[level] * len(transformed_data), cmap='viridis')
        ax.set_xlabel('FPCA Component 1')
        ax.set_ylabel('FPCA Component 2')
        ax.set_title(f'FPCA of ABR waveforms for cluster {cluster_id}')
        plt.colorbar(scatter, label="dB Level")
        plt.show()

    return fpca_results


kMeansClusterPCA = apply_fpca_on_clusters(kmeans_dict, abr_list, level=105, n_components=20)

DbscanPCAClusters = apply_fpca_on_clusters(dbscan_dict, abr_list, level=105, n_components=20)

def plot_fpca_cluster_analysis(level, cluster_id, abr_list, fpca_cluster_results, cluster_membership_dict, eigenvecNo=3):
    """
    Plots FPCA analysis for a specific cluster at a given level.
    Includes eigenvectors, component distribution, and extreme waveform examples.
    """
    # Retrieve FPCA results
    cluster_results = fpca_cluster_results[cluster_id]
    fpca = cluster_results['fpca']
    transformed_data = cluster_results['transformed_data']
    explained_variance = cluster_results['explained_variance']

    # Get waveform indices for this cluster
    if cluster_id not in cluster_membership_dict:
        print(f"Cluster {cluster_id} not found in the cluster membership dictionary.")
        return

    cluster_indices = cluster_membership_dict[cluster_id]

    # # ✨ NEW: Filter out-of-bounds indices
    # max_index = transformed_data.shape[0]
    # original_len = len(cluster_indices)
    # cluster_indices = [i for i in cluster_indices if i < max_index]
    # if len(cluster_indices) < original_len:
    #     print(f"[Warning] Skipped {original_len - len(cluster_indices)} out-of-bounds indices in cluster {cluster_id}")


    # Get original waveforms at this level
    abr_wave = next(wave for wave, lvl in abr_list if lvl == level)
    waveforms = np.array(abr_wave)

    # Extract only the waveforms in this cluster
    cluster_waveforms = waveforms[cluster_indices]
    time_points = np.linspace(0, 1, cluster_waveforms.shape[1])

    ## 1️⃣ Plot Eigenvectors (FPCA Components)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i in range(min(eigenvecNo, len(fpca.components_))):
        ax1.plot(time_points, fpca.components_[i].data_matrix[0, :, 0], label=f'Component {i+1} ({explained_variance[i]*100:.2f}%)')

    ax1.set_title(f'FPCA Components - Cluster {cluster_id} ({level} dB)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    plt.show()

    ## 2️⃣ Plot FPCA Score Distribution with Extreme Points
    cluster_transformed = transformed_data #[cluster_indices]
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    scatter = ax2.scatter(cluster_transformed[:, 0], cluster_transformed[:, 1], alpha=0.6)

    extremes = {
        'top': np.argmax(cluster_transformed[:, 1]),
        'bottom': np.argmin(cluster_transformed[:, 1]),
        'right': np.argmax(cluster_transformed[:, 0]),
        'left': np.argmin(cluster_transformed[:, 0])
    }

    for direction, local_idx in extremes.items():
        ax2.annotate(str(local_idx), (cluster_transformed[local_idx, 0], cluster_transformed[local_idx, 1]),
                     textcoords="offset points", xytext=(0,10 if direction == 'top' else -15), ha='center')

    ax2.set_title(f'FPCA Component Distribution - Cluster {cluster_id} ({level} dB)\nExtreme Points Numbered')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    plt.show()

    ## 3️⃣ Plot Extreme Waveforms (Original vs Reconstructed)
    fig3, axs = plt.subplots(2, 2, figsize=(15, 10))
    extreme_indices = list(extremes.values())

    for i, local_idx in enumerate(extreme_indices[:4]):
        ax = axs[i//2, i%2]
        original = cluster_waveforms[local_idx]
        reconstructed = fpca.inverse_transform(transformed_data[local_idx:local_idx+1]).data_matrix[0, :, 0]

        # try:
        #     reconstructed = fpca.inverse_transform(transformed_data[global_idx:global_idx+1]).data_matrix[0, :, 0]
        #     ax.plot(time_points, original, label='Original')
        #     ax.plot(time_points, reconstructed, '--', label='Reconstructed')
        # except (IndexError, AttributeError, ValueError) as e:
        #     print(f"[Warning] Skipping reconstruction for index {global_idx}: {e}")
        #     ax.plot(time_points, original, label='Original (Reconstruction Failed)', color='red')

        ax.plot(time_points, original, label='Original')
        ax.plot(time_points, reconstructed, '--', label='Reconstructed')
        ax.set_title(f'Extreme {local_idx} - {list(extremes.keys())[i]}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    plt.show()

    ## 5️⃣ Plot All Waveforms in the Cluster (Stacked)
    plt.figure(figsize=(12, 6))
    for i, waveform in enumerate(cluster_waveforms):
        plt.plot(time_points, waveform, alpha=0.3, color='tab:blue')  # light lines for stacking effect

    plt.title(f'All Waveforms in Cluster {cluster_id} at {level} dB')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_fpca_cluster_analysis(
    level=105.0,
    cluster_id=1,
    abr_list=abr_list,
    fpca_cluster_results=kMeansClusterPCA,
    cluster_membership_dict=kmeans_dict
)

plot_fpca_cluster_analysis(
    level=105.0,
    cluster_id=2,
    abr_list=abr_list,
    fpca_cluster_results=kMeansClusterPCA,
    cluster_membership_dict=kmeans_dict
)

plot_fpca_cluster_analysis(
    level=105.0,
    cluster_id=0,
    abr_list=abr_list,
    fpca_cluster_results= DbscanPCAClusters,
    cluster_membership_dict=dbscan_dict
)

def Complete_FPCA_Analysis(abr_data, abr_thresholds, frequency=4000, n_components=5, n_groups=3, level=105.0):
    """
    Complete FPCA analysis pipeline including clustering and visualization.
    """
    # Step 1: Apply FPCA on the entire dataset
    fpca_results = apply_fpca_per_level(abr_data, abr_thresholds, frequency, n_components, selected_level=level)

    # Step 2: Cluster the FPCA results using KMeans
    kmeans_dict = cluster_and_plot_kmeans(level, abr_list, fpca_results, n_groups=n_groups)

    # Step 3: Apply FPCA on each cluster
    kMeansClusterPCA = apply_fpca_on_clusters(kmeans_dict, abr_list, level=105, n_components=n_components)

    # Step 4: Plot FPCA analysis for each cluster
    for cluster_id in kMeansClusterPCA.keys():
        plot_fpca_cluster_analysis(
            level=level,
            cluster_id=cluster_id,
            abr_list=abr_list,
            fpca_cluster_results=kMeansClusterPCA,
            cluster_membership_dict=kmeans_dict
        )
    # Step 5: Cluster the FPCA results using DBSCAN
    dbscan_dict = cluster_and_plot_dbscan(level, abr_list, fpca_results, eps=0.5*(1e-5), min_samples=5)
    # Step 6: Apply FPCA on each cluster
    DbscanPCAClusters = apply_fpca_on_clusters(dbscan_dict, abr_list, level=105, n_components=n_components)
    # Step 7: Plot FPCA analysis for each cluster
    for cluster_id in DbscanPCAClusters.keys():
        plot_fpca_cluster_analysis(
            level=level,
            cluster_id=cluster_id,
            abr_list=abr_list,
            fpca_cluster_results=DbscanPCAClusters,
            cluster_membership_dict=dbscan_dict
        )
    return fpca_results, kMeansClusterPCA, DbscanPCAClusters

fpca_1, k_fpca1, d_fpca1 = Complete_FPCA_Analysis(
    abr_data,
    abr_thresholds,
    frequency=32000,
    n_components=20,
    n_groups=3,
    level=105.0
    )

fpca_1, k_fpca1, d_fpca1 = Complete_FPCA_Analysis(
    abr_data,
    abr_thresholds,
    frequency=16000,
    n_components=20,
    n_groups=3,
    level=105.0
    )

def filter_abr_by_threshold(abr_list, crop_range=(0, 1), threshold=1.5):
    """
    Applies threshold-based filtering to ABR data grouped by level.
    
    Returns:
    - filtered_data: List of tuples (filtered_waveforms, mean_waveform, level)
    """
    filtered_data = []
    for group, level in abr_list:
        filtered_waveforms, mean_waveform = process_waveforms(
            group, crop_range=crop_range, threshold=threshold
        )
        filtered_data.append((filtered_waveforms, mean_waveform, level))
    return filtered_data


def filter_abr_by_ci(abr_list, crop_range=(0, 1), ci=95):
    """
    Applies confidence-interval-based filtering to ABR data grouped by level.

    Returns:
    - filtered_data: List of tuples (filtered_waveforms, mean_waveform, level)
    """
    filtered_data = []
    for group, level in abr_list:
        filtered_waveforms, mean_waveform = process_waveforms_ci(
            group, crop_range=crop_range, ci=ci
        )
        filtered_data.append((filtered_waveforms, mean_waveform, level))
    return filtered_data


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

mouse_id = 89
time_point = 0
ear = 'right'
frequency = 8000

filename = f"~/data/practicum-data/ABRpresto data/ABRpresto data csv/Mouse{mouse_id}_timepoint{time_point}_{ear} abr_io {frequency}.csv" 

abr_single_trial_data = pd.read_csv(filename, index_col=[0, 1, 2])

abr_single_trial_data.head()

float_cols = abr_single_trial_data.select_dtypes(include=['float']).columns
new_column_names = {col: i + 1 for i, col in enumerate(float_cols)}
abr_single_trial_data.rename(columns=new_column_names, inplace=True)

abr_single_trial_data.head(5)

grouped = abr_single_trial_data.groupby(level=[0])

type(grouped)



# def process_waveforms(group, crop_range=(0, 1), threshold=1.5):
#     """
#     Process waveforms by filtering extreme rows and cropping the time range.
    
#     Args:
#     - group (np.ndarray): 2D array of shape (n_waveforms, n_samples) containing the waveforms.
#     - crop_range (tuple): The normalized start and end range for cropping (e.g., (0.2, 0.8)).
#     - threshold (float): Multiplier to filter waveforms beyond the threshold of the max mean of absolute values.
    
#     Returns:
#     - cropped_filtered_waveforms (np.ndarray): The processed and cropped waveforms.
#     - mean_waveform (np.ndarray): The mean waveform after filtering and cropping.
#     """
#     # Step 1: Calculate the mean of the absolute values across all waveforms
#     abs_mean_waveform = np.mean(np.abs(group), axis=0)
#     abs_extreme_value = np.max(abs_mean_waveform)  # Maximum of the mean absolute values
    
#     # Step 2: Filter rows (waveforms)
#     if threshold is None:
#         filtered_waveforms = group
#     else:
#         mask = np.all(np.abs(group) <= threshold * abs_extreme_value, axis=1)
#         filtered_waveforms = group[mask]  # Keep only rows where all values are within the threshold
    
#     # Step 3: Handle case where no rows remain after filtering
#     if filtered_waveforms.size == 0:
#         filtered_waveforms = group

#     # Step 4: Recalculate mean after filtering
#     mean_waveform = np.mean(filtered_waveforms, axis=0)
    
#     # Step 5: Crop the waveforms based on the crop range
#     if crop_range is None:
#         return filtered_waveforms, mean_waveform
#     n_samples = mean_waveform.shape[0]
#     start_idx = int(crop_range[0] * n_samples)
#     end_idx = int(crop_range[1] * n_samples)
    
#     cropped_filtered_waveforms = filtered_waveforms[:, start_idx:end_idx]
    
#     # Step 6: Calculate the mean waveform after cropping
#     mean_waveform = np.mean(cropped_filtered_waveforms, axis=0)
    
#     return cropped_filtered_waveforms, mean_waveform


from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import numpy as np

def process_waveforms(group, crop_range=(0, 1), threshold=1.5, target_length=244, interpolate=True):
    """
    Process waveforms by filtering extreme rows, cropping the time range, and smoothing.
    
    Args:
    - group (np.ndarray): 2D array of shape (n_waveforms, n_samples) containing the waveforms.
    - crop_range (tuple): The normalized start and end range for cropping (e.g., (0.2, 0.8)).
    - threshold (float): Multiplier to filter waveforms beyond the threshold of the max mean of absolute values.
    - target_length (int): The desired length for the interpolated waveforms.
    - interpolate (bool): Whether to interpolate the waveforms to the target length.
    
    Returns:
    - interpolated_filtered_waveforms (np.ndarray): The filtered, cropped, interpolated, and smoothed waveforms.
    - smoothed_mean_waveform (np.ndarray): The interpolated and smoothed mean waveform.
    """
    # Step 1: Calculate the mean of the absolute values across all waveforms
    abs_mean_waveform = np.mean(np.abs(group), axis=0)
    abs_extreme_value = np.max(abs_mean_waveform)  # Maximum of the mean absolute values
    
    # Step 2: Filter rows (waveforms)
    if threshold is None:
        filtered_waveforms = group
    else:
        mask = np.all(np.abs(group) <= threshold * abs_extreme_value, axis=1)
        filtered_waveforms = group[mask]  # Keep only rows where all values are within the threshold
    
    # Step 3: Handle case where no rows remain after filtering
    if filtered_waveforms.size == 0:
        filtered_waveforms = group

    # Step 4: Recalculate mean after filtering
    mean_waveform = np.mean(filtered_waveforms, axis=0)
    
    # Step 5: Crop the waveforms based on the crop range
    if crop_range is not None:
        n_samples = mean_waveform.shape[0]
        start_idx = int(crop_range[0] * n_samples)
        end_idx = int(crop_range[1] * n_samples)
        filtered_waveforms = filtered_waveforms[:, start_idx:end_idx]
        mean_waveform = mean_waveform[start_idx:end_idx]

    if not interpolate:
        return filtered_waveforms, mean_waveform
    
    # Step 6: Interpolate and smooth the filtered waveforms
    interpolated_filtered_waveforms = np.array(
        [interpolate_and_smooth(waveform, target_length) for waveform in filtered_waveforms]
    )
    
    # Step 7: Interpolate and smooth the mean waveform
    smoothed_mean_waveform = interpolate_and_smooth(mean_waveform, target_length=target_length)
    
    return interpolated_filtered_waveforms, smoothed_mean_waveform

def interpolate_and_smooth(waveform, target_length=244, smoothing_sigma=1):
    """
    Interpolate and smooth a waveform to a target length.
    
    Args:
    - waveform (np.ndarray): 1D array representing the waveform.
    - target_length (int): The desired length for the output waveform.
    - smoothing_sigma (float): Standard deviation for Gaussian smoothing.
    
    Returns:
    - smoothed_waveform (np.ndarray): The interpolated and smoothed waveform.
    """
    # Interpolation using cubic spline
    original_indices = np.arange(len(waveform))
    target_indices = np.linspace(0, len(waveform) - 1, target_length)
    cs = CubicSpline(original_indices, waveform)
    interpolated_waveform = cs(target_indices)
    
    # Apply Gaussian smoothing
    smoothed_waveform = gaussian_filter1d(interpolated_waveform, sigma=smoothing_sigma)
    
    return smoothed_waveform


def process_waveforms_ci(group, crop_range=(0, 1), ci=95, target_length=244, interpolate=True):
    """
    Filter waveforms using 95% confidence interval bounds at each time point
    """
    # Calculate initial percentiles
    lower_bound, upper_bound = np.percentile(group, [(100-ci)/2, 100-(100-ci)/2], axis=0)
    
    # Filter waveforms that stay within bounds at all time points
    mask = np.all((group >= lower_bound) & (group <= upper_bound), axis=1)
    filtered_waveforms = group[mask]
    
    # Fallback if empty
    if filtered_waveforms.size == 0:
        filtered_waveforms = group.copy()

    # Crop processing
    if crop_range is not None:
        n_samples = filtered_waveforms.shape[1]
        start = int(crop_range[0] * n_samples)
        end = int(crop_range[1] * n_samples)
        filtered_waveforms = filtered_waveforms[:, start:end]

    # Calculate final mean
    mean_waveform = np.mean(filtered_waveforms, axis=0)

    # Interpolation and smoothing
    if interpolate:
        processed = np.array([interpolate_and_smooth(wf, target_length) for wf in filtered_waveforms])
        mean_waveform = interpolate_and_smooth(mean_waveform, target_length)
        return processed, mean_waveform
    
    return filtered_waveforms, mean_waveform

def process_waveforms_cosine(group, crop_range=(0, 1), similarity_thresh=0.999, target_length=244, interpolate=True):
    """
    Filter waveforms based on cosine similarity to median waveform
    """
    # Calculate reference waveform
    mean_wf = np.mean(group, axis=0)
    
    # Calculate similarities
    similarities = cosine_similarity(group, mean_wf.reshape(1, -1)).flatten()
    # print("Cosine Similarities:", similarities)
    
    # Apply threshold
    mask = similarities >= similarity_thresh
    # print("Length of true values in mask:", len(mask[mask == True]))
    # print("Length of false values in mask:", len(mask[mask == False]))
    filtered_waveforms = group[mask]
    
    # Fallback if empty
    if filtered_waveforms.size == 0:
        filtered_waveforms = group.copy()

    # Crop processing
    if crop_range is not None:
        n_samples = filtered_waveforms.shape[1]
        start = int(crop_range[0] * n_samples)
        end = int(crop_range[1] * n_samples)
        filtered_waveforms = filtered_waveforms[:, start:end]

    # Calculate final mean
    mean_waveform = np.mean(filtered_waveforms, axis=0)

    # Interpolation and smoothing
    if interpolate:
        processed = np.array([interpolate_and_smooth(wf, target_length) for wf in filtered_waveforms])
        mean_waveform = interpolate_and_smooth(mean_waveform, target_length)
        return processed, mean_waveform
    
    return filtered_waveforms, mean_waveform

def process_waveforms_svd(group, crop_range=(0, 1), n_components=3, target_length=244, interpolate=True):
    """
    Filter waveforms using SVD reconstruction error
    """
    # Perform SVD decomposition
    svd = TruncatedSVD(n_components=n_components)
    reduced = svd.fit_transform(group)
    reconstructed = svd.inverse_transform(reduced)
    
    # Calculate reconstruction errors
    errors = np.linalg.norm(group - reconstructed, axis=1)

    error_thresh = np.percentile(errors, 20)
    
    # Apply error threshold
    mask = errors <= error_thresh
    filtered_waveforms = group[mask]
    
    # Fallback if empty
    if filtered_waveforms.size == 0:
        filtered_waveforms = group.copy()

    # Crop processing
    if crop_range is not None:
        n_samples = filtered_waveforms.shape[1]
        start = int(crop_range[0] * n_samples)
        end = int(crop_range[1] * n_samples)
        filtered_waveforms = filtered_waveforms[:, start:end]

    # Calculate final mean
    mean_waveform = np.mean(filtered_waveforms, axis=0)

    # Interpolation and smoothing
    if interpolate:
        processed = np.array([interpolate_and_smooth(wf, target_length) for wf in filtered_waveforms])
        mean_waveform = interpolate_and_smooth(mean_waveform, target_length)
        return processed, mean_waveform
    
    return filtered_waveforms, mean_waveform

mean_waveforms = []
for decibel, group in grouped:
    data = group.values  # Extract readings
    # Plot individual readings
    # plt.figure(figsize=(12, 8))
    # for reading in data:
    #     plt.plot(reading, alpha=0.3, label='_nolegend_')  # Transparency for individual plots
    
    # Calculate and plot the mean
    cropped_filtered_waveforms, mean_waveform = process_waveforms(
    data, 
    crop_range=None, 
    threshold=2,
    interpolate=True
    )

    mean_waveforms.append(mean_waveform)

    plt.figure(figsize=(12, 8))
    for reading in cropped_filtered_waveforms:
        plt.plot(reading, alpha=0.3, label='_nolegend_') 

    # mean_reading = np.mean(data, axis=0)
    plt.plot(mean_waveform, color='red', linewidth=2, label='Mean Reading')

    # Add titles and labels
    plt.title(f"Readings for Decibel: {decibel}")
    plt.xlabel("Time (samples)")  # Replace with appropriate x-axis label
    plt.ylabel("Amplitude (scaled)")  # Replace with appropriate y-axis label
    plt.legend()
    plt.show()

mean_waveforms_ci = []
for decibel, group in grouped:
    data = group.values  # Extract readings
    # Plot individual readings
    # plt.figure(figsize=(12, 8))
    # for reading in data:
    #     plt.plot(reading, alpha=0.3, label='_nolegend_')  # Transparency for individual plots
    
    # Calculate and plot the mean
    cropped_filtered_waveforms, mean_waveform = process_waveforms_ci(
    data, 
    crop_range=None, 
    ci=97.5,
    interpolate=True
    )
    mean_waveforms_ci.append(mean_waveform)

    plt.figure(figsize=(12, 8))
    for reading in cropped_filtered_waveforms:
        plt.plot(reading, alpha=0.3, label='_nolegend_') 

    # mean_reading = np.mean(data, axis=0)
    plt.plot(mean_waveform, color='red', linewidth=2, label='Mean Reading')

    # Add titles and labels
    plt.title(f"Readings for Decibel: {decibel}")
    plt.xlabel("Time (samples)")  # Replace with appropriate x-axis label
    plt.ylabel("Amplitude (scaled)")  # Replace with appropriate y-axis label
    plt.legend()
    plt.show()

mean_waveforms_cos = []
for decibel, group in grouped:
    data = group.values  # Extract readings
    # Plot individual readings
    # plt.figure(figsize=(12, 8))
    # for reading in data:
    #     plt.plot(reading, alpha=0.3, label='_nolegend_')  # Transparency for individual plots
    
    # Calculate and plot the mean
    cropped_filtered_waveforms, mean_waveform = process_waveforms_cosine(
    data, 
    crop_range=None, 
    similarity_thresh=0.40,
    interpolate=True
    )

    mean_waveforms_cos.append(mean_waveform)

    plt.figure(figsize=(12, 8))
    for reading in cropped_filtered_waveforms:
        plt.plot(reading, alpha=0.3, label='_nolegend_') 

    # mean_reading = np.mean(data, axis=0)
    plt.plot(mean_waveform, color='red', linewidth=2, label='Mean Reading')

    # Add titles and labels
    plt.title(f"Readings for Decibel: {decibel}")
    plt.xlabel("Time (samples)")  # Replace with appropriate x-axis label
    plt.ylabel("Amplitude (scaled)")  # Replace with appropriate y-axis label
    plt.legend()
    plt.show()

mean_waveforms_svd = []
for decibel, group in grouped:
    data = group.values  # Extract readings
    # Plot individual readings
    # plt.figure(figsize=(12, 8))
    # for reading in data:
    #     plt.plot(reading, alpha=0.3, label='_nolegend_')  # Transparency for individual plots
    
    # Calculate and plot the mean
    cropped_filtered_waveforms, mean_waveform = process_waveforms_svd(
    data, 
    crop_range=None, 
    n_components=5,
    interpolate=True
    )

    mean_waveforms_svd.append(mean_waveform)

    plt.figure(figsize=(12, 8))
    for reading in cropped_filtered_waveforms:
        plt.plot(reading, alpha=0.3, label='_nolegend_') 

    # mean_reading = np.mean(data, axis=0)
    plt.plot(mean_waveform, color='red', linewidth=2, label='Mean Reading')

    # Add titles and labels
    plt.title(f"Readings for Decibel: {decibel}")
    plt.xlabel("Time (samples)")  # Replace with appropriate x-axis label
    plt.ylabel("Amplitude (scaled)")  # Replace with appropriate y-axis label
    plt.legend()
    plt.show()

plt.figure(figsize=(12, 8))
for reading in mean_waveforms:
    plt.plot(reading, alpha=0.3, label='_nolegend_') 

# Add titles and labels
plt.title(f"Waves using thresholding function")
plt.xlabel("Time (samples)")  # Replace with appropriate x-axis label
plt.ylabel("Amplitude (scaled)")  # Replace with appropriate y-axis label
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for reading in mean_waveforms_ci:
    plt.plot(reading, alpha=0.3, label='_nolegend_') 

# Add titles and labels
plt.title(f"Readings for mouse:{mouse_id}, timepoint:{time_point}, ear:{ear}, frequency:{frequency}")
plt.xlabel("Time (samples)")  # Replace with appropriate x-axis label
plt.ylabel("Amplitude (scaled)")  # Replace with appropriate y-axis label
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for reading in mean_waveforms_cos:
    plt.plot(reading, alpha=0.3, label='_nolegend_') 

# Add titles and labels
plt.title(f"Readings for mouse:{mouse_id}, timepoint:{time_point}, ear:{ear}, frequency:{frequency}")
plt.xlabel("Time (samples)")  # Replace with appropriate x-axis label
plt.ylabel("Amplitude (scaled)")  # Replace with appropriate y-axis label
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for reading in mean_waveforms_svd:
    plt.plot(reading, alpha=0.3, label='_nolegend_') 

# Add titles and labels
plt.title(f"Readings for mouse:{mouse_id}, timepoint:{time_point}, ear:{ear}, frequency:{frequency}")
plt.xlabel("Time (samples)")  # Replace with appropriate x-axis label
plt.ylabel("Amplitude (scaled)")  # Replace with appropriate y-axis label
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# List of decibels to plot together
decibels_to_plot = [35, 75, 105]

# Create subplots with a number of columns equal to the number of decibels to plot
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

for i, decibel in enumerate(decibels_to_plot):
    # Get the group for the current decibel level
    group = grouped.get_group(decibel)
    data = group.values  # Extract the readings
    
    # Process the waveforms
    cropped_filtered_waveforms, mean_waveform = process_waveforms(
        data, crop_range=None, threshold=2, interpolate=True
    )
    
    # Plot individual waveforms with transparency on the current subplot
    # for reading in cropped_filtered_waveforms:
    #     axes[i].plot(reading, alpha=0.3, label='_nolegend_')  # Transparency for individual plots
    
    # Plot the mean waveform in red
    axes[i].plot(mean_waveform, color='red', linewidth=2, label='Mean Reading')

    # Add title, labels, and legend
    axes[i].set_title(f"Readings for {decibel} dB")
    axes[i].set_xlabel("Time (samples)")  # Replace with appropriate x-axis label
    axes[i].set_ylabel("Amplitude (scaled)")  # Replace with appropriate y-axis label
    axes[i].legend()

# Adjust layout for better spacing between subplots
plt.tight_layout()
plt.show()


import plotly.graph_objects as go
import numpy as np

def plot_waves_with_plotly(mouse_id, time_point, ear, frequency, mean_waveforms):
    # Initialize the figure
    fig = go.Figure()

    # Add each waveform to the figure
    for i, waveform in enumerate(mean_waveforms):
        # Plot each waveform with a slight transparency
        fig.add_trace(go.Scatter(
            x=np.linspace(0, len(waveform), len(waveform)), 
            y=waveform,
            mode='lines',
            line=dict(color=f'rgba(0, 0, 0, 0.3)'),  # Set transparency using RGBA
            name=f"Waveform {i+1}",
            showlegend=False  # Hide individual legends for waveforms
        ))

    # Add title and labels
    fig.update_layout(
        title=f"Readings for Mouse: {mouse_id}, Timepoint: {time_point}, Ear: {ear}, Frequency: {frequency}",
        xaxis_title="Time (samples)",  # Replace with appropriate x-axis label
        yaxis_title="Amplitude (scaled)",  # Replace with appropriate y-axis label
        width=1000,
        height=600,
        showlegend=True  # Show legend for the waveforms
    )

    # Show the figure
    fig.show()

# Example usage
plot_waves_with_plotly(mouse_id='M001', time_point='T1', ear='Left', frequency=1000, mean_waveforms=mean_waveforms)


import matplotlib.pyplot as plt

db_list = list(range(15, 106, 5))

# Create a figure for the stacked plot
plt.figure(figsize=(12, 16))

# Stack the waveforms by adding an offset to each
offset = 0  # Initial offset (very small)
for i, reading in enumerate(mean_waveforms_ci):
    # Skip if the decibel level is not in the specified levels
    if db_list[i] not in [35, 45, 55, 65, 75, 85, 95, 105]:
        continue  # Skip the current iteration if the condition is not met
    
    # Plot each waveform with an increasing offset
    plt.plot(reading + offset, alpha=0.3, label=f"{db_list[i]} dB")  
    offset += 5 * (1e-6)  # Increment the offset to stack waveforms on top of each other

# Add titles and labels
plt.title(f"Waveforms using confidence interval function")

# Hide the y-axis
plt.gca().get_yaxis().set_visible(False)

# Remove the ticks on the y-axis
plt.yticks([])

# Show the x-axis ticks and labels
plt.xlabel("Time (samples)")  # Or use another label if you prefer

# Show the plot
plt.legend()
plt.show()


import matplotlib.pyplot as plt

mean_waveforms = []
for decibel, group in grouped:
    data = group.values  # Extract readings
    
    # # Process the waveforms
    # cropped_filtered_waveforms, mean_waveform = process_waveforms(
    #     data, 
    #     crop_range=None, 
    #     threshold=2,
    #     interpolate=True
    # )

    # cropped_filtered_waveforms, mean_waveform = process_waveforms_cosine(
    # data, 
    # crop_range=None, 
    # similarity_thresh=0.50,
    # interpolate=True
    # )

    cropped_filtered_waveforms, mean_waveform = process_waveforms_svd(
    data, 
    crop_range=None, 
    n_components=5,
    interpolate=True
    )

    mean_waveforms.append(mean_waveform)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))  # 2 rows, 1 column

    # Plot cropped_filtered_waveforms in the first subplot
    for reading in cropped_filtered_waveforms:
        ax1.plot(reading, alpha=0.3, label='_nolegend_')
    ax1.set_title(f"Cropped and Filtered Waveforms for Decibel: {decibel}")
    ax1.set_xlabel("Time (samples)")
    ax1.set_ylabel("Amplitude (scaled)")
    ax1.legend()

    # Plot the original data and the mean waveform in the second subplot
    for reading in data:
        ax2.plot(reading, alpha=0.3, label='_nolegend_')
    ax2.set_title(f"Original Readings for Decibel: {decibel}")
    ax2.set_xlabel("Time (samples)")
    ax2.set_ylabel("Amplitude (scaled)")
    ax2.legend()

    # Show the plots
    plt.tight_layout()  # Adjust subplots to fit into figure area
    plt.show()


