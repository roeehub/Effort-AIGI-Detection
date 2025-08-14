import pandas as pd
import os
import random
import subprocess
from tqdm import tqdm

# --- Configuration ---
# 1. Set the paths to your CSV files.
video_csv_path = '/Users/adammokiy/Documents/Dtect_Work/codes/youtube_dataset.csv'
image_csv_path = '/Users/adammokiy/Documents/Dtect_Work/codes/utk_dataset.csv'

# 2. Set the base paths for your datasets.
video_base_path = '/Users/adammokiy/Documents/Dtect_Work/load_to_buckets'
image_base_path = '/Users/adammokiy/Documents/Dtect_Work/load_to_buckets'

# 3. Set the base output directory for the generated deepfakes.
output_base_path = '/Users/adammokiy/Documents/Dtect_Work/load_to_buckets/deep_cam_dataset'

# 4. Set the path to the Deep-Live-Cam execution script.
#    (e.g., 'run.py' or '/path/to/Deep-Live-Cam/run.py')
deep_live_cam_script_path = 'run.py'

# 5. Set the number of pairs you want to generate.
NUM_PAIRS_TO_CREATE = 3000

# --- Main Script ---

def create_deepfake_dataset():
    """
    Main function to generate the deepfake dataset.
    """
    # --- Step 1: Load and Filter Data ---
    print("Step 1: Loading and filtering data...")
    try:
        # Load the video dataset (AVSpeech)
        video_df = pd.read_csv(video_csv_path, header=None, names=['path', 'age', 'gender', 'race'])
        
        # Load the image dataset (UTK)
        image_df = pd.read_csv(image_csv_path)
    except FileNotFoundError as e:
        print(f"Error: CSV file not found. {e}")
        print("Please ensure your CSV file paths are correct in the configuration section.")
        return

     # ##################################################################################
    # ### CHANGE STARTS HERE: Convert 'race' and 'gender' to lowercase for consistency ###
    # ##################################################################################
    # This ensures that values like 'White' and 'white' are treated as the same category,
    # which is crucial for the grouping step to work correctly.
    # We also use .str.strip() to remove any accidental leading/trailing whitespace.

    print("Normalizing 'race' and 'gender' columns to lowercase...")
    for col in ['race', 'gender']:
        video_df[col] = video_df[col].str.strip().str.lower()
        image_df[col] = image_df[col].str.strip().str.lower()
    
    # ########################
    # ### CHANGE ENDS HERE ###
    # ########################

    # Convert 'age' columns to a numeric type.
    video_df['age'] = pd.to_numeric(video_df['age'], errors='coerce')
    image_df['age'] = pd.to_numeric(image_df['age'], errors='coerce')

    # Remove rows where the age could not be converted (i.e., where age is now NaN).
    video_df.dropna(subset=['age'], inplace=True)
    image_df.dropna(subset=['age'], inplace=True)

    # Now that the data is clean, convert the 'age' column to integer.
    video_df['age'] = video_df['age'].astype(int)
    image_df['age'] = image_df['age'].astype(int)
    # --- FIX ENDS HERE ---

    # Filter both dataframes for ages between 18 and 50.
    video_df_filtered = video_df[(video_df['age'] >= 18) & (video_df['age'] <= 50)].copy()
    image_df_filtered = image_df[(image_df['age'] >= 18) & (image_df['age'] <= 50)].copy()
    print(f"Found {len(video_df_filtered)} valid video identities and {len(image_df_filtered)} valid image identities.")

    # --- Step 2: Group Data by Race and Gender ---
    print("\nStep 2: Grouping data by race and gender...")
    video_groups = video_df_filtered.groupby(['race', 'gender'])
    image_groups = image_df_filtered.groupby(['race', 'gender'])

    # --- Step 3: Create Random Pairs ---
    print("\nStep 3: Creating random pairs...")
    all_possible_pairs = []
    
    # Find common (race, gender) groups between the two datasets.
    common_groups = set(video_groups.groups.keys()) & set(image_groups.groups.keys())

    for group in common_groups:
        race, gender = group
        video_identities = video_groups.get_group(group)['path'].tolist()
        image_identities = image_groups.get_group(group)['path'].tolist()

        # Shuffle lists to ensure random pairing.
        random.shuffle(video_identities)
        random.shuffle(image_identities)

        # Pair up individuals from the same group.
        num_pairs_in_group = min(len(video_identities), len(image_identities))
        for i in range(num_pairs_in_group):
            # Each pair is (source_image_path, target_video_path)
            all_possible_pairs.append((image_identities[i], video_identities[i]))
            
    if not all_possible_pairs:
        print("Error: No common individuals found that match the criteria. Exiting.")
        return

    # Check if we can create the desired number of pairs.
    if len(all_possible_pairs) < NUM_PAIRS_TO_CREATE:
        print(f"Warning: Only {len(all_possible_pairs)} unique pairs could be formed. Proceeding with this number.")
        num_to_select = len(all_possible_pairs)
    else:
        num_to_select = NUM_PAIRS_TO_CREATE

    # Randomly select the final set of pairs.
    selected_pairs = random.sample(all_possible_pairs, num_to_select)
    print(f"Successfully selected {len(selected_pairs)} pairs for deepfake generation.")

    # --- Step 4: Process Pairs and Generate Deepfakes ---
    print("\nStep 4: Starting deepfake generation process...")
    os.makedirs(output_base_path, exist_ok=True)

    for i, (image_relative_path, video_relative_path) in enumerate(selected_pairs):
        print(f"\n--- Processing Pair {i+1}/{len(selected_pairs)} ---")
        print(f"Source (Image): {image_relative_path}")
        print(f"Target (Video): {video_relative_path}")

        # Construct full paths for source and target.
        source_image_full_path = os.path.join(image_base_path, image_relative_path)
        target_video_dir_full_path = os.path.join(video_base_path, video_relative_path)

        # Create a dedicated output folder for the current video's deepfakes.
        video_folder_name = os.path.basename(video_relative_path)
        output_video_folder = os.path.join(output_base_path, video_folder_name)
        os.makedirs(output_video_folder, exist_ok=True)

        # Validate that the source image and target directory exist.
        if not os.path.isfile(source_image_full_path):
            tqdm.write(f"  [SKIP] Pair {i+1}: Source image not found at: {source_image_full_path}")
            continue
        if not os.path.isdir(target_video_dir_full_path):
            tqdm.write(f"  [SKIP] Pair {i+1}: Target video directory not found at: {target_video_dir_full_path}")
            continue

        # Iterate over all .jpg frames in the target video folder.
        try:
            frames = sorted([f for f in os.listdir(target_video_dir_full_path) if f.lower().endswith('.jpg')])
            if not frames:
                tqdm.write(f"  [WARNING] Pair {i+1}: No .jpg frames found in {target_video_dir_full_path}")
                continue

            for frame_filename in tqdm(frames):
                target_frame_full_path = os.path.join(target_video_dir_full_path, frame_filename)
                output_frame_full_path = os.path.join(output_video_folder, frame_filename)

                # Construct the command for Deep-Live-Cam.
                command = [
                    'python',
                    deep_live_cam_script_path,
                    '--source', source_image_full_path,
                    '--target', target_frame_full_path,
                    '--output', output_frame_full_path,
                    '--execution-provider', 'coreml'
                ]

                print(f"  - Generating frame: {frame_filename}...")
                
                # Execute the deepfake generation command.
                # Using capture_output to hide command's stdout unless an error occurs.
                
                ### CHANGE START
                # Run and capture both stdout & stderr
                result = subprocess.run(command, capture_output=True, text=True)

                # Show output for debugging
                print(result.stdout)
                print(result.stderr)

                # Merge both outputs for easier searching
                combined_output = (result.stdout or "") + (result.stderr or "")

                # Look for the specific face detection failure message
                if "Face detection failed for target or source" in combined_output:
                    tqdm.write(f"  [SKIP] Pair {i+1}: Face detection failed on frame {frame_filename}.")
                    
                    # Delete the output folder for this video
                    shutil.rmtree(output_video_folder, ignore_errors=True)
                    tqdm.write(f"  [CLEANUP] Deleted output folder: {output_video_folder}")
                    
                    tqdm.write("Moving to next video pair...")
                    break  # break the frame loop

                # If command failed with another error
                if result.returncode != 0:
                    tqdm.write(f"  [ERROR] Deep-Live-Cam failed for pair {i+1} ({video_folder_name}).")
                    tqdm.write(f"    - Output: {combined_output.strip()}")
                    
                    # Cleanup on error too
                    shutil.rmtree(output_video_folder, ignore_errors=True)
                    tqdm.write(f"  [CLEANUP] Deleted output folder: {output_video_folder}")
                    
                    break
                ### CHANGE END

        except FileNotFoundError:
            tqdm.write(f"  [ERROR] Command not found. Ensure Python is installed and '{deep_live_cam_script_path}' exists.")
            break 
        except subprocess.CalledProcessError as e:
            # Check if the error is the specific face detection failure.
            # Check if the error is the specific face detection failure.
            if 'Face detection failed for target or source' in e.stderr:
                # Print a specific message indicating we are skipping this pair.
                tqdm.write(f"  [SKIP] Pair {i+1}: Face detection failed. Continuing to the next pair.")
                
            else:
                # If it's a different error, print the full details for debugging.
                tqdm.write(f"  [ERROR] Deep-Live-Cam failed for pair {i+1} ({video_folder_name}).")
                tqdm.write(f"    - Stderr: {e.stderr.strip()}")

            # Use 'continue' to skip the rest of this pair's processing
            # and move to the next iteration of the main loop.
            tqdm.write("Moving To Next Video Folder")
            continue

            # In either case of a CalledProcessError, we break from the frame loop 
            # and move to the next video pair.
            break
        except Exception as e:
            tqdm.write(f"  [ERROR] An unexpected error occurred on pair {i+1}: {e}")
            continue

    print("\n--- Deepfake generation process completed. ---")


if __name__ == "__main__":
    # A note on performance
    print("####################################################################")
    print("# NOTE: This script will execute a computationally intensive task. #")
    print("# Generating deepfakes for many videos will take a long time.    #")
    print("####################################################################\n")
    
    create_deepfake_dataset()