import logging
import requests  # noqa
import time
from typing import List, Dict, Tuple

import pandas as pd  # noqa
from google.cloud import storage  # noqa
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix  # noqa
from tqdm import tqdm  # noqa

# --- CONFIGURATION ---
# API endpoint URL provided in the prompt
API_BASE_URL = "http://34.118.121.111:8999"
# GCS bucket and folder containing the test data
BUCKET_NAME = "deep-fake-test-10-08-25"
GCS_DATA_PREFIX = "Deep fake test 10.08.25/"
# Models to evaluate
MODEL_TYPES = ["base", "custom"]
# Log file for recording errors and results
LOG_FILE = "evaluation_log.txt"
# Request timeout in seconds
REQUEST_TIMEOUT = 120  # Increased timeout for potentially long video processing

# --- LOGGING SETUP ---
# Set up logging to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)


def get_videos_from_gcs(bucket_name: str, prefix: str) -> List[Dict[str, str]]:
    """
    Lists all video files in a GCS bucket folder and extracts their paths and labels.

    Args:
        bucket_name: The name of the GCS bucket.
        prefix: The folder path inside the bucket to search for videos.

    Returns:
        A list of dictionaries, where each dictionary contains the 'gcs_path'
        and the ground truth 'label' ('real' or 'fake').
    """
    logging.info(f"Attempting to list videos from bucket '{bucket_name}' with prefix '{prefix}'...")
    videos_to_test = []
    try:
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

        for blob in blobs:
            # Skip non-video files or directory placeholders
            if not blob.name.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                continue

            if "/fake/" in blob.name:
                label = "fake"
            elif "/real/" in blob.name:
                label = "real"
            else:
                logging.warning(f"Could not determine label for {blob.name}. Skipping.")
                continue

            videos_to_test.append({"gcs_path": blob.name, "label": label})

    except Exception as e:
        logging.error(f"FATAL: Failed to list files from GCS bucket '{bucket_name}'.")
        logging.error(f"Please ensure you have authenticated with 'gcloud auth application-default login'.")
        logging.error(f"Error details: {e}")
        return []

    logging.info(f"Found {len(videos_to_test)} videos to test.")
    return videos_to_test


def run_evaluation(api_url: str, videos: List[Dict[str, str]], model_type: str) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Sends requests to the API for each video and collects the results.

    Args:
        api_url: The base URL of the detection API.
        videos: A list of videos with their GCS paths and true labels.
        model_type: The model type to use ('base' or 'custom').

    Returns:
        A tuple containing a pandas DataFrame with the results and a list of errors.
    """
    endpoint = f"{api_url}/check_video_from_gcp"
    results = []
    errors = []

    logging.info(f"--- Starting evaluation for model_type: '{model_type}' ---")
    for video in tqdm(videos, desc=f"Testing '{model_type}' model"):
        full_gcs_path = f"{BUCKET_NAME}/{video['gcs_path']}"
        payload = {"gcs_path": full_gcs_path}
        params = {
            "model_type": model_type,
            "return_probs": "false",  # Not needed for evaluation, reduces response size
            "debug": "false"
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            # Check for HTTP errors
            if response.status_code != 200:
                error_detail = response.json().get('detail', response.text)
                logging.warning(
                    f"API Error for {video['gcs_path']} (HTTP {response.status_code}): {error_detail}"
                )
                errors.append({"gcs_path": video['gcs_path'], "error": error_detail})
                continue

            # Process successful response
            data = response.json()
            results.append({
                "gcs_path": video['gcs_path'],
                "true_label": video['label'],
                "pred_label": data['pred_label'].lower(),
                "fake_prob": data['fake_prob']
            })

        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {video['gcs_path']}: {e}")
            errors.append({"gcs_path": video['gcs_path'], "error": str(e)})

        # A small delay to avoid overwhelming the API endpoint
        time.sleep(0.1)

    return pd.DataFrame(results), errors


def print_metrics(model_type: str, results_df: pd.DataFrame):
    """Calculates and logs standard classification metrics."""
    if results_df.empty:
        logging.warning(f"No results to evaluate for model '{model_type}'.")
        return

    logging.info(f"--- Performance Metrics for model_type: '{model_type}' ---")

    # Map labels to binary for scikit-learn functions
    y_true = results_df['true_label'].map({'real': 0, 'fake': 1})
    y_pred = results_df['pred_label'].map({'real': 0, 'fake': 1})
    y_pred_prob = results_df['fake_prob']
    labels = ["real", "fake"]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError as e:
        auc = f"Not computable: {e}"  # Handle cases with only one class present in results

    report = classification_report(y_true, y_pred, target_names=labels)
    cm = confusion_matrix(y_true, y_pred)

    # Log metrics
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Weighted F1-Score: {f1:.4f}")
    logging.info(f"AUC Score: {auc if isinstance(auc, str) else f'{auc:.4f}'}")
    logging.info("\n" + report)
    logging.info("Confusion Matrix:")
    logging.info(f"\n{pd.DataFrame(cm, index=[f'True {l}' for l in labels], columns=[f'Pred {l}' for l in labels])}")
    logging.info("-" * 50)


def main():
    """Main script execution function."""
    logging.info("Starting API Evaluation Script")

    videos_to_test = get_videos_from_gcs(BUCKET_NAME, GCS_DATA_PREFIX)
    if not videos_to_test:
        logging.error("Halting script as no videos could be retrieved from GCS.")
        return

    all_errors = {}
    for model in MODEL_TYPES:
        results_df, errors = run_evaluation(API_BASE_URL, videos_to_test, model)
        all_errors[model] = errors

        if not results_df.empty:
            print_metrics(model, results_df)
        else:
            logging.warning(f"Evaluation skipped for model '{model}' due to lack of successful API responses.")

    # --- Final Summary ---
    logging.info("--- Final Error Summary ---")
    for model, errors in all_errors.items():
        if errors:
            logging.warning(f"Model '{model}' encountered {len(errors)} errors (videos skipped):")
            for i, error in enumerate(errors, 1):
                logging.warning(f"  {i}. Video: {error['gcs_path']} | Reason: {error['error']}")
        else:
            logging.info(f"Model '{model}' ran without any processing errors.")
    logging.info(f"Evaluation finished. Full details logged to '{LOG_FILE}'.")


if __name__ == "__main__":
    main()

"""### How to Run the Script

1.  Save the code above into a file named `evaluate_api.py`.
2.  Make sure you have completed the steps in the **Prerequisites** section.
3.  Open your terminal, navigate to the directory where you saved the file, and run it:

    ```bash
    python evaluate_api.py
    ```

### What to Expect

The script will perform the following actions:

1.  **Log Initialization:** It will create a file named `evaluation_log.txt` in the same directory to store all output. It will also print this output to your console.
2.  **GCS File Listing:** It will connect to your GCS bucket and list all the video files, showing you a count of how many it found.
3.  **Model Evaluation:** For each model (`base` and `custom`), it will display a progress bar (`tqdm`) as it sends requests to the API.
    *   Any videos that result in an error (e.g., face not found, server error) will be logged and skipped.
4.  **Metrics Reporting:** After each model is tested, it will print a detailed report including:
    *   Accuracy, F1-Score, and AUC.
    *   A full classification report with precision and recall for both "real" and "fake" classes.
    *   A confusion matrix to show true vs. predicted classifications.
5.  **Final Summary:** At the end, it will provide a summary of all the videos that were skipped due to errors for each model, making it easy to identify problematic files."""
