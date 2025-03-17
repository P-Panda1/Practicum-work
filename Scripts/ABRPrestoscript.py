import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

# Define directories
input_directory = os.path.expanduser(
    '~/data/practicum-data/ABRpresto data/ABRpresto data csv/')
output_directory = os.path.expanduser(
    '~/data/practicum-data/ABRpresto data/processed_data/')
os.makedirs(output_directory, exist_ok=True)


def interpolate_and_smooth(waveform, target_length=244, smoothing_sigma=1):
    """
    Interpolate and smooth a waveform to a target length.
    """
    original_indices = np.arange(len(waveform))
    target_indices = np.linspace(0, len(waveform) - 1, target_length)
    cs = CubicSpline(original_indices, waveform)
    interpolated_waveform = cs(target_indices)
    smoothed_waveform = gaussian_filter1d(
        interpolated_waveform, sigma=smoothing_sigma)
    return smoothed_waveform


def process_waveforms(group, crop_range=(0, 1), threshold=2, target_length=244, interpolate=True):
    """
    Process waveforms with filtering, cropping, interpolation, and smoothing.
    """
    # Calculate mean of absolute values across waveforms
    abs_mean_waveform = np.mean(np.abs(group), axis=0)
    abs_extreme_value = np.max(abs_mean_waveform)

    # Filter waveforms
    mask = np.all(np.abs(group) <= threshold * abs_extreme_value, axis=1)
    filtered_waveforms = group[mask]

    # Fallback if all waveforms are filtered
    if filtered_waveforms.size == 0:
        filtered_waveforms = group

    # Crop waveforms
    if crop_range is not None:
        n_samples = filtered_waveforms.shape[1]
        start_idx = int(crop_range[0] * n_samples)
        end_idx = int(crop_range[1] * n_samples)
        filtered_waveforms = filtered_waveforms[:, start_idx:end_idx]

    # Interpolate and smooth
    if interpolate:
        processed_waveforms = np.array([interpolate_and_smooth(
            wf, target_length) for wf in filtered_waveforms])
        mean_waveform = interpolate_and_smooth(
            np.mean(filtered_waveforms, axis=0), target_length)
    else:
        processed_waveforms = filtered_waveforms
        mean_waveform = np.mean(filtered_waveforms, axis=0)

    return processed_waveforms, mean_waveform


# Initialize progress tracking
file_count = len([f for f in os.listdir(
    input_directory) if f.endswith(".csv")])
processed_count = 0
processed_data = []

for file_name in os.listdir(input_directory):
    if file_name.endswith(".csv"):
        print(f"Processing {processed_count + 1}/{file_count}: {file_name}")
        try:
            file_path = os.path.join(input_directory, file_name)
            base_name = os.path.splitext(file_name)[0]
            parts = base_name.split('_')

            # Extract metadata
            mouse_id = int(parts[0].replace("Mouse", ""))
            timepoint = int(parts[1].replace("timepoint", ""))
            ear = 'right' if 'right' in parts[2].lower() else 'left'
            frequency = int(parts[-1].split()[-1])

            # Read data and process
            data = pd.read_csv(file_path).drop(columns=['t0'])
            grouped = data.groupby(['level', 'polarity'])

            for (level, polarity), group in grouped:
                # Convert to numpy array and process
                waveforms = group.drop(
                    columns=['level', 'polarity']).astype(float).to_numpy()
                _, mean_waveform = process_waveforms(
                    waveforms, crop_range=(0, 1), threshold=1.5)

                # Store results
                waveform_dict = {i+1: val for i,
                                 val in enumerate(mean_waveform)}
                processed_data.append({
                    'id': mouse_id,
                    'timepoint': timepoint,
                    'ear': ear,
                    'frequency': frequency,
                    'level': level,
                    'polarity': polarity,
                    **waveform_dict
                })

            processed_count += 1
            print(f"Processed: {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

# Save processed data
processed_df = pd.DataFrame(processed_data)
output_path = os.path.join(output_directory, 'processed_waveforms.csv')
processed_df.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")
