import os
import cv2
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

def analyze_video_and_create_csv(base_directory, output_csv_path):
    """
    Finds all .mp4 files with 'real' in their path within a directory,
    analyzes the age, gender, and race from a frame of each video using DeepFace,
    and saves the results to a CSV file.

    Args:
        base_directory (str): The absolute path to the directory to search.
        output_csv_path (str): The path to save the output CSV file.
    """
    video_files_to_process = []
    # Recursively find all .mp4 files that contain 'real' in their path
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.mp4'):
                full_path = os.path.join(root, file)
                # if 'real' in full_path:
                # Get the relative path from the 'vfhq' directory
                relative_path = os.path.relpath(full_path, os.path.dirname(base_directory))
                video_files_to_process.append({'path': relative_path, 'full_path': full_path})

    # if not video_files_to_process:
    #     print("No .mp4 files with 'real' in their path were found.")
    #     return

    print(f"Found {len(video_files_to_process)} videos to analyze.")

    results = []
    for video_info in tqdm(video_files_to_process):
        video_path = video_info['full_path']
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue

        # Get a frame from the middle of the video for analysis
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Could not read frame from video: {video_path}")
            continue

        # Analyze the frame using DeepFace
        # The 'enforce_detection' parameter is set to False to avoid errors if no face is detected
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=['age', 'gender', 'race'],
            enforce_detection=False
        )

        # Check if a face was detected and analyzed
        if isinstance(analysis, list) and len(analysis) > 0:
            person_data = analysis[0]
            race_predictions = person_data.get('race', {})
            # Find the dominant race
            dominant_race = max(race_predictions, key=race_predictions.get) if race_predictions else 'N/A'
            
            # Format the race output
            race_mapping = {
                'asian': 'Asian',
                'white': 'White',
                'black': 'Black',
                'Other': 'Other'
            }
            formatted_race = race_mapping.get(dominant_race.lower(), 'Others')
            
            gender_data = person_data.get('gender', 'N/A')
            if isinstance(gender_data, dict):
                # Get the key (gender) with the max probability
                dominant_gender = max(gender_data, key=gender_data.get)
                # Map keys to your preferred labels
                gender_map = {'man': 'male', 'woman': 'female'}
                gender_label = gender_map.get(dominant_gender.lower(), 'unknown')
            else:
                # If already a string, just normalize it
                gender_label = str(gender_data).lower()
                if gender_label in ['man', 'male']:
                    gender_label = 'male'
                elif gender_label in ['woman', 'female']:
                    gender_label = 'female'
                else:
                    gender_label = 'unknown'

            results.append({
                'path': video_info['path'],
                'age': person_data.get('age', 'N/A'),
                'gender': gender_label,
                'race': formatted_race
            })
        else:
            print(f"No face detected or analysis failed for video: {video_path}")
            results.append({
                'path': video_info['path'],
                'age': 'N/A',
                'gender': 'N/A',
                'race': 'N/A'
            })

        
    
    # Rename columns as per the user's request
    df = pd.DataFrame(results)
    df.rename(columns={'gender': 'gender', 'race': 'race'}, inplace=True)
    df['gender'] = df['gender'].str.lower()
    
    # Custom column names for age are not standard for a CSV. 
    # The age will be in a single column named 'age'.
    # If you need to have columns 0, 1, 2, 3... for age, it would imply a different data structure,
    # which is not ideal for this kind of analysis. The current format is standard.
    
    # Save the results to a CSV file
    df.to_csv(output_csv_path, index=False)
    # print(f"Analysis complete. Results saved to {output_csv_path}")

if __name__ == '__main__':
    # The directory containing the 'vfhq' folder
    # IMPORTANT: Change this to the directory that CONTAINS your video dataset folder
    # For example, if your path is /Users/adammokiy/Documents/thesis/data/vfhq,
    # then the base_dir should be /Users/adammokiy/Documents/thesis/data
    base_dir = '/Users/adammokiy/Documents/Dtect_Work/load_to_buckets/youtube_dataset'

    # The name of the output CSV file
    output_csv = '/Users/adammokiy/Documents/Dtect_Work/codes/youtube_demographics.csv'

    analyze_video_and_create_csv(base_dir, output_csv)