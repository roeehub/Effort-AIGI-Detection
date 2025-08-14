
import os
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm



def analyze_video_frames(base_directory):
    """
    Analyzes the first frame of each video folder in a directory to detect age,
    gender, and race using the DeepFace library.

    Args:
        base_directory (str): The absolute path to the directory containing the video folders.

    Returns:
        pandas.DataFrame: A DataFrame containing the relative path, age, gender, and race
                          for each video folder where a face was detected.
    """
    results = []
    # Ensure the base directory is in the correct format
    base_directory = os.path.normpath(base_directory)
    # Get the parent directory of the base_directory for creating relative paths
    parent_directory = os.path.dirname(base_directory)


    if not os.path.isdir(base_directory):
        print(f"Error: Directory not found at {base_directory}")
        return pd.DataFrame()

    video_folders = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    for video_folder in tqdm(video_folders):
        video_folder_path = os.path.join(base_directory, video_folder)
        
        try:
            # Find the first frame file (assuming JPEG format)
            frames = sorted([f for f in os.listdir(video_folder_path) if f.lower().endswith('.jpg')])
            if not frames:
                print(f"No JPEG frames found in {video_folder_path}")
                continue

            first_frame_path = os.path.join(video_folder_path, frames[0])
            relative_path = os.path.relpath(video_folder_path, parent_directory)


            # Analyze the first frame for age, gender, and race
            analysis = DeepFace.analyze(
                img_path=first_frame_path,
                actions=['age', 'gender', 'race'],
                enforce_detection=True  # Ensure a face is detected
            )
            
            # The result from DeepFace is a list of dictionaries, one for each detected face.
            # We will process the first detected face.
            if isinstance(analysis, list) and len(analysis) > 0:
                face_data = analysis[0]
                
                # Categorize race
                dominant_race = face_data.get('dominant_race', 'unknown')
                if dominant_race.lower() in ['white', 'black', 'asian']:
                    race = dominant_race.lower()
                else:
                    race = 'other'

                # Append the results
                results.append({
                    'relative_path': relative_path,
                    'age': face_data.get('age', 'N/A'),
                    'gender': face_data.get('gender', 'N/A'),
                    'race': race
                })
            else:
                 print(f"Could not extract face data object from analysis for {first_frame_path}")


        except Exception as e:
            print(f"Could not analyze {video_folder}: {e}")

    return pd.DataFrame(results)

if __name__ == '__main__':
    # The main directory containing the video folders
    main_directory = '/Users/adammokiy/Documents/Dtect_Work/load_to_buckets/external_youtube_avspeech'

    # Analyze the frames and get the results
    demographics_df = analyze_video_frames(main_directory)

    if not demographics_df.empty:
        # Save the DataFrame to a CSV file
        output_csv_path = 'demographics.csv'
        demographics_df.to_csv(output_csv_path, index=False)
        print(f"Analysis complete. Results saved to {output_csv_path}")
    else:
        print("No data was processed to save.")