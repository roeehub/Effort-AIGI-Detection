import re
from google.cloud import storage

# --- Configuration ---
# Set the specific bucket and prefix you want to scan.
BUCKET_NAME = 'deep-fake-test-10-08-25'
PREFIX_TO_SCAN = 'Deep fake test 10.08.25/fake/veo3/'
OUTPUT_FILE = 'veo3_test_ids.txt'


def extract_ids_from_prefix(storage_client, bucket_name, prefix):
    """
    Scans a specific GCS prefix, extracts all potential numeric IDs from
    the filenames, and returns them as a set.
    """
    print(f"--- Starting Scan ---")
    print(f"Bucket: gs://{bucket_name}")
    print(f"Prefix: {prefix}\n")

    found_ids = set()
    file_count = 0
    bucket = storage_client.bucket(bucket_name)

    # This regex will find any sequence of 10 or more digits.
    # It's robust enough to find IDs like '10052349848554105426'.
    id_pattern = re.compile(r'(\d{10,})')

    # List all files (blobs) under the specified prefix
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        # Ignore any "folder" objects that might be listed
        if blob.name.endswith('/'):
            continue

        file_count += 1

        # Find all numeric sequences in the filename that match the pattern
        matches = id_pattern.findall(blob.name)
        if matches:
            # Add all found ID strings to our set of unique IDs
            found_ids.update(matches)
        else:
            print(f"[Warning] No ID pattern found in filename: {blob.name}")

    print(f"\nScanned a total of {file_count} files in the prefix.")
    return found_ids


def main():
    """Main function to run the ID extraction."""
    try:
        storage_client = storage.Client()
    except Exception as e:
        print(f"Error initializing Google Cloud Storage client: {e}")
        print("Please ensure you are authenticated with 'gcloud auth application-default login'")
        return

    # 1. Extract the IDs from the specified GCS path
    extracted_ids = extract_ids_from_prefix(storage_client, BUCKET_NAME, PREFIX_TO_SCAN)

    # 2. Report and save the results
    print(f"Found {len(extracted_ids)} unique potential IDs.")

    if extracted_ids:
        print(f"Saving extracted IDs to: {OUTPUT_FILE}")

        # Sort the IDs for a clean, predictable output file
        sorted_ids = sorted(list(extracted_ids))

        with open(OUTPUT_FILE, 'w') as f:
            for video_id in sorted_ids:
                f.write(f"{video_id}\n")

        print(f"--- Complete ---")
        print(f"Successfully saved {len(sorted_ids)} IDs to {OUTPUT_FILE}.")
    else:
        print("--- Complete ---")
        print("No IDs matching the pattern were found in the specified path.")


if __name__ == "__main__":
    main()
