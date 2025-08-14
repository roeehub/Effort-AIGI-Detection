import pandas as pd
import os
import random
import subprocess
from tqdm import tqdm

def create_deepfake_dataset_unique(source_dataset_csv, target_video_analysis_csv, output_dir, max_pairs=1000):
    """
    Creates a deepfake dataset by creating unique 1-to-1 pairs of sources and targets,
    ensuring no source image or target video is used more than once.

    Args:
        source_dataset_csv (str): The file path for the UTK dataset CSV.
        target_video_analysis_csv (str): The file path for the video analysis results CSV.
        output_dir (str): The directory where the output deepfakes will be saved.
        max_pairs (int): The maximum number of unique deepfake pairs to generate.
    """
    # --- Define Base Paths ---
    base_utk_path = '/Users/adammokiy/Documents/Dtect_Work/load_to_buckets'
    base_vfhq_path = '/Users/adammokiy/Documents/Dtect_Work/load_to_buckets'
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # --- Load and Preprocess Datasets ---
    print("Loading and preprocessing datasets...")
    utk_df = pd.read_csv(source_dataset_csv)
    utk_df['path'] = utk_df['path'].apply(lambda p: os.path.join(base_utk_path, p))

    video_df = pd.read_csv(target_video_analysis_csv)
    
    def resolve_video_path(p):
        if pd.isna(p): return None
        p = str(p)
        if os.path.isabs(p): return p
        if p.startswith('youtube_dataset/'): return os.path.join(base_vfhq_path, p)
        return p

    video_df['path'] = video_df['path'].apply(resolve_video_path)
    video_df.dropna(subset=['path'], inplace=True)

    # --- Filter by Age ---
    print("Filtering datasets by age (18-50)...")
    utk_df = utk_df[(utk_df['age'] >= 18) & (utk_df['age'] <= 50)]
    video_df = video_df[(video_df['age'] >= 18) & (video_df['age'] <= 50)]

    if utk_df.empty or video_df.empty:
        print("One of the dataframes is empty after age filtering. Exiting.")
        return

    # --- Create Unique 1-to-1 Pairs within each group ---
    print("Identifying all possible unique 1-to-1 pairs...")
    utk_grouped = utk_df.groupby(['gender', 'race'])
    video_grouped = video_df.groupby(['gender', 'race'])
    all_possible_pairs = []

    # Find common groups (intersections of gender/race in both datasets)
    common_groups = set(utk_grouped.groups.keys()) & set(video_grouped.groups.keys())

    for group_key in common_groups:
        utk_group = utk_grouped.get_group(group_key)
        video_group = video_grouped.get_group(group_key)

        source_paths = utk_group['path'].tolist()
        target_paths = video_group['path'].tolist()

        # Shuffle both lists to ensure random pairing
        random.shuffle(source_paths)
        random.shuffle(target_paths)

        # Determine the number of pairs we can make (the smaller of the two lists)
        num_pairs_in_group = min(len(source_paths), len(target_paths))

        # Create unique pairs
        for i in range(num_pairs_in_group):
            all_possible_pairs.append((source_paths[i], target_paths[i]))

    if not all_possible_pairs:
        print("No valid pairs found matching the criteria (same gender/race, age 18-50).")
        return

    # --- Select a Random Subset of the Unique Pairs ---
    print(f"Found {len(all_possible_pairs)} total unique 1-to-1 pairs.")
    random.shuffle(all_possible_pairs) # Shuffle the master list of pairs
    selected_pairs = all_possible_pairs[:max_pairs]
    print(f"Processing a random selection of {len(selected_pairs)} unique pairs...")

    # --- Iterate, and Execute ---
    for i, (source_path, target_path) in tqdm(enumerate(selected_pairs)):
        print(f"\n--- Processing Pair {i+1} of {len(selected_pairs)} ---")
        print(f"Source: {source_path}")
        print(f"Target: {target_path}")

        try:
            target_basename = os.path.basename(str(target_path))
            
            if str(target_path).endswith('.mp4'):
                output_filename = f"deepfake_{target_basename}"
                output_path = os.path.join(output_dir, output_filename)
                command = ['python', 'run.py', '--source', source_path, '--target', target_path, '--output', output_path, '--execution-provider', 'coreml']
                print(f"Executing: {' '.join(command)}")
                subprocess.run(command, check=True)

            else:
                if not os.path.isdir(target_path):
                    print(f"Warning: Target path {target_path} is not a directory. Skipping.")
                    continue
                    
                video_folder_name = os.path.basename(target_path)
                output_video_dir = os.path.join(output_dir, video_folder_name)
                if not os.path.exists(output_video_dir): os.makedirs(output_video_dir)

                for frame_filename in os.listdir(target_path):
                    if frame_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        frame_path = os.path.join(target_path, frame_filename)
                        output_frame_path = os.path.join(output_video_dir, frame_filename)
                        command = ['python', 'run.py', '--source', source_path, '--target', frame_path, '--output', output_frame_path, '--execution-provider', 'coreml']
                        print(f"Executing for frame: {' '.join(command)}")
                        subprocess.run(command, check=True)
        
        except Exception as e:
            print(f"An error occurred while processing pair ({source_path}, {target_path}): {e}")

    print(f"\nScript finished. Processed {len(selected_pairs)} unique pairs.")


if __name__ == '__main__':
    source_dataset_csv = '/Users/adammokiy/Documents/Dtect_Work/codes/utk_dataset.csv'
    target_video_analysis_csv = '/Users/adammokiy/Documents/Dtect_Work/codes/youtube_demographics.csv'
    output_directory = '/Users/adammokiy/Documents/Dtect_Work/load_to_buckets/deep_cam_dataset'

    create_deepfake_dataset_unique(source_dataset_csv, target_video_analysis_csv, output_directory, max_pairs=1000)