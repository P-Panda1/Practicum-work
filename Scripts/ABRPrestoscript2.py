import os
import pandas as pd
from tqdm import tqdm  # For progress tracking
from cftsdata import abr

# Define paths
source_dir = os.path.abspath(
    "../../../data/practicum-data/ABRpresto data/ABRpresto data")
destination_dir = os.path.abspath(
    "../../data/practicum-data/ABRpresto data/processed_data")
output_file = os.path.join(destination_dir, "updated_processed_waveforms.csv")

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Initialize list to store processed data
all_data = []

# Get list of folders in the source directory
folders = [f for f in os.listdir(
    source_dir) if os.path.isdir(os.path.join(source_dir, f))]


def extract_metadata(folder_name):
    """Extract mouse_id, time_point, and ear from folder name."""
    parts = folder_name.split('_')
    mouse_id = parts[0].replace('Mouse', '')
    time_point = parts[1].replace('timepoint', '')
    ear = parts[2]  # Assumes ear is always in the third part
    return mouse_id, time_point, ear


# Process each folder with progress tracking
for folder in tqdm(folders, desc="Processing Folders"):
    folder_path = os.path.join(source_dir, folder)
    mouse_id, time_point, ear = extract_metadata(folder)

    try:
        # Print folder name for debugging
        print(f"Processing folder: {folder}")

        # Load ABR data
        try:
            fh = abr.load(folder_path)
        except Exception as e:
            print(f"Skipping {folder} due to error in abr.load: {e}")
            continue

        try:
            epochs = fh.get_epochs_filtered()
            if epochs.empty:
                raise ValueError("Epochs data is empty")
            epochs_mean = epochs.groupby(['frequency', 'level']).mean()
        except Exception as e:
            print(
                f"Skipping {folder} due to error in get_epochs_filtered: {e}")
            continue

        # Process each frequency-level pair
        for (freq, level), group in epochs_mean.iterrows():
            wave_data = group.values  # Replace with actual waveform column if needed

            # Convert wave array into separate columns
            wave_dict = {f'wave_{i+1}': val for i, val in enumerate(wave_data)}

            # Append to list
            all_data.append({
                'mouse_id': mouse_id,
                'time_point': time_point,
                'ear': ear,
                'frequency': freq,
                'level': level,
                **wave_dict
            })
    except Exception as e:
        print(f"Error processing {folder}: {e}")

# Create DataFrame and save as CSV
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Processing complete. CSV saved to {output_file}")
else:
    print("No data was processed. CSV was not created.")
