import os
import csv

def create_utk_csv_robust(dataset_path):
    """
    Creates a CSV file for the UTKFace dataset, handling inconsistencies
    in filenames.

    The CSV file will contain the relative path to each image, and the age,
    gender, and race extracted from the filename. It gracefully handles
    files where the race code might be missing.

    Args:
        dataset_path (str): The full path to the UTKFace dataset directory.
    """
    # Define the mapping for gender and race
    gender_map = {0: 'male', 1: 'female'}
    race_map = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}

    # Define the path for the output CSV file
    # This will save the CSV in the same directory where the script is run.
    output_csv_path = 'utk_dataset.csv'

    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['path', 'age', 'gender', 'race'])

        print(f"Starting to process files in: {dataset_path}")
        processed_count = 0
        skipped_count = 0

        for filename in os.listdir(dataset_path):
            if filename.lower().endswith('.jpg'):
                # Clean up the filename by removing the extension for easier parsing
                name_parts = filename.replace('.jpg', '').split('_')

                # A valid name must have at least age and gender.
                if len(name_parts) < 2:
                    print(f"Skipping malformed filename (not enough parts): {filename}")
                    skipped_count += 1
                    continue

                try:
                    # The first two parts should always be age and gender
                    age = int(name_parts[0])
                    gender_code = int(name_parts[1])

                    # The third part is race, but it might be missing
                    # If there are 4 parts, we assume the 3rd is race
                    # e.g., [age]_[gender]_[race]_[datetime]
                    if len(name_parts) >= 4:
                        race_code = int(name_parts[2])
                    # If there are 3 parts, the race is missing
                    # e.g., [age]_[gender]_[datetime]
                    else:
                        race_code = None # No race information available

                    # Convert codes to human-readable strings
                    gender = gender_map.get(gender_code, 'unknown')
                    # Use 'unknown' if race_code is None or not in the map
                    race = race_map.get(race_code, 'unknown') if race_code is not None else 'unknown'

                    # Check for invalid gender codes that might actually be race codes
                    if gender == 'unknown':
                        print(f"Skipping file with invalid gender code: {filename}")
                        skipped_count += 1
                        continue

                    # Get the relative path to the image
                    relative_path = os.path.join('UTK', filename)

                    # Write the data to the CSV file
                    csv_writer.writerow([relative_path, age, gender, race])
                    processed_count += 1

                except ValueError:
                    # This catches errors if age, gender, or race cannot be converted to int
                    print(f"Skipping malformed filename (value error): {filename}")
                    skipped_count += 1
                    continue

    print("\n--- Processing Complete ---")
    print(f"Successfully processed {processed_count} files.")
    print(f"Skipped {skipped_count} malformed or problematic files.")
    print(f"CSV file created successfully at: {os.path.abspath(output_csv_path)}")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Replace this with the actual path to your UTK dataset directory
    utk_dataset_directory = '/Users/adammokiy/Documents/Dtect_Work/load_to_buckets/UTK'

    if os.path.isdir(utk_dataset_directory):
        create_utk_csv_robust(utk_dataset_directory)
    else:
        print(f"Error: The specified directory does not exist: {utk_dataset_directory}")
        print("Please update the 'utk_dataset_directory' variable with the correct path.")