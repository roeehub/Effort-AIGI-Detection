"""
GCS (Google Cloud Storage) utilities for downloading assets.

This module provides functions to download checkpoints, models, and other
assets from GCS buckets during training initialization.

Usage:
    from utils.gcs import download_assets_from_gcs
    
    local_paths = download_assets_from_gcs(config, logger)
    if local_paths:
        checkpoint_path = local_paths['base_checkpoint']
"""

import os
import time
import traceback
from typing import Optional, List, Dict

from google.cloud import storage
from google.api_core import exceptions
from google.cloud.storage import Bucket, Blob


def download_blob_streaming(blob: Blob, destination_path: str, logger) -> bool:
    """
    Download a blob using streaming to avoid loading entire file into memory.
    
    Uses the GCS Python SDK's download_to_filename which streams to disk.
    This is more memory-efficient than download_as_bytes for large files.
    
    Args:
        blob: The GCS blob to download
        destination_path: Local path to save the file
        logger: Logger instance
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get file size for logging
        blob.reload()
        file_size = blob.size
        file_size_mb = file_size / (1024 * 1024) if file_size else 0
        
        logger.info(f"    File size: {file_size_mb:.1f} MB - streaming download to disk")
        
        # download_to_filename streams to disk, doesn't load into memory
        # Set a longer timeout for large files (30 min)
        blob.download_to_filename(destination_path, timeout=1800)
        
        logger.info(f"    ✅ Streaming download complete")
        return True
        
    except Exception as e:
        logger.error(f"    Streaming download failed: {e}")
        logger.error(f"    Traceback: {traceback.format_exc()}")
        return False


# Threshold for using streaming download for large files (100MB)
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB


def download_gcs_asset(
    bucket: Bucket,
    gcs_path: str,
    local_path: str,
    logger,
    allowed_files: Optional[List[str]] = None
) -> bool:
    """
    Downloads a single blob or a directory of blobs from GCS.

    Args:
        bucket (storage.Bucket): The GCS bucket object.
        gcs_path (str): The path to the object or directory in GCS.
        local_path (str): The local path to download to.
        logger: The logger instance.
        allowed_files: Optional list of specific files to download from a directory.

    Returns:
        bool: True if successful, False otherwise.
    """
    if gcs_path.endswith('/'):  # It's a directory
        prefix = gcs_path.split(bucket.name + '/', 1)[1]
        logger.info(f"  Downloading directory with prefix: '{prefix}'")
        os.makedirs(local_path, exist_ok=True)

        if allowed_files:
            normalized_paths = [p.lstrip('/') for p in allowed_files]
            logger.info(f"  Downloading {len(normalized_paths)} specific files: {normalized_paths}")
            for rel_path in normalized_paths:
                blob_name = f"{prefix}{rel_path}" if prefix else rel_path
                gcs_uri = f"gs://{bucket.name}/{blob_name}"
                logger.info(f"  Checking if blob exists: {gcs_uri}")
                blob = bucket.blob(blob_name)
                try:
                    blob.reload()  # Get metadata including size
                    file_size = blob.size
                    file_size_mb = file_size / (1024 * 1024) if file_size else 0
                    logger.info(f"    File size: {file_size_mb:.1f} MB")
                except Exception as e:
                    logger.error(f"Error checking blob for {gcs_uri}: {e}")
                    logger.error(f"  Traceback: {traceback.format_exc()}")
                    return False

                destination_file_name = os.path.join(local_path, rel_path)
                os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
                logger.info(f"  Downloading: {blob_name} -> {destination_file_name}")
                
                try:
                    # Use streaming download for large files (streams directly to disk)
                    if file_size and file_size > LARGE_FILE_THRESHOLD:
                        if not download_blob_streaming(blob, destination_file_name, logger):
                            return False
                    else:
                        blob.download_to_filename(destination_file_name)
                    logger.info(f"  ✅ Downloaded: {rel_path}")
                except Exception as e:
                    logger.error(f"Failed to download {blob.name}: {e}")
                    logger.error(f"  Traceback: {traceback.format_exc()}")
                    return False

            logger.info(
                f"Downloaded {len(normalized_paths)} specific file(s) from {gcs_path}")
            return True

        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            logger.error(f"Directory {gcs_path} is empty or does not exist.")
            return False
            
        downloaded = False
        blob_count = 0
        for blob in blobs:
            if blob.name.endswith('/'):  # Skip "directory" blobs
                continue
            blob_count += 1
            destination_file_name = os.path.join(local_path, os.path.relpath(blob.name, prefix))
            logger.info(f"  Downloading blob: {blob.name} -> {destination_file_name}")
            os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
            try:
                # Use streaming download for large files
                if blob.size and blob.size > LARGE_FILE_THRESHOLD:
                    if not download_blob_streaming(blob, destination_file_name, logger):
                        return False
                else:
                    blob.download_to_filename(destination_file_name)
                downloaded = True
            except Exception as e:
                logger.error(f"Failed to download {blob.name}: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
                return False
        if not downloaded:
            logger.error(f"Directory {gcs_path} has no files (only subdirectories).")
            return False
        logger.info(f"  Downloaded {blob_count} files from directory.")
        return True
    else:  # It's a single file
        blob_name = gcs_path.split(bucket.name + '/', 1)[1]
        blob = bucket.blob(blob_name)
        try:
            blob.reload()
        except Exception as e:
            logger.error(f"File not found at {gcs_path}: {e}")
            return False
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # Use streaming download for large files
        if blob.size and blob.size > LARGE_FILE_THRESHOLD:
            return download_blob_streaming(blob, local_path, logger)
        blob.download_to_filename(local_path)
        return True


def download_assets_from_gcs(config: dict, logger) -> Optional[Dict[str, str]]:
    """
    Downloads specified assets (checkpoints, models) from a GCS bucket.

    This function reads a list of assets from the config, where each asset has
    a GCS path and a desired local path. It handles both individual files and
    entire directories.

    Args:
        config (dict): The main configuration dictionary containing 'gcs_assets' key.
        logger: The logger instance for logging messages.

    Returns:
        dict: A dictionary mapping asset keys to their local paths if successful,
              otherwise None.
    
    Example config structure:
        gcs_assets:
          base_checkpoint:
            gcs_path: gs://bucket/path/to/checkpoint.pth
            local_path: ./weights/checkpoint.pth
          clip_backbone:
            gcs_path: gs://bucket/path/to/model/
            local_path: ./weights/clip/
            files: ["config.json", "pytorch_model.bin"]  # Optional: specific files
    """
    assets_to_download = config.get('gcs_assets')
    if not assets_to_download:
        logger.info("No GCS assets configured for download. Skipping.")
        return None

    local_paths = {}

    # First, check if all assets already exist locally
    all_exist = True
    for key, asset_info in assets_to_download.items():
        local_path = asset_info.get('local_path')
        if not local_path or not os.path.exists(local_path):
            all_exist = False
            break
    if all_exist:
        logger.info("All GCS assets already exist locally. Skipping downloads.")
        for key, asset_info in assets_to_download.items():
            local_paths[key] = asset_info.get('local_path')
        return local_paths

    logger.info("--- GCS Asset Download ---")
    try:
        storage_client = storage.Client()
        start_time = time.time()

        for key, asset_info in assets_to_download.items():
            gcs_path = asset_info.get('gcs_path')
            local_path = asset_info.get('local_path')

            if not gcs_path or not local_path:
                logger.error(f"Asset '{key}' is missing 'gcs_path' or 'local_path' in config.")
                return None

            if not gcs_path.startswith('gs://'):
                logger.error(f"Invalid GCS path for asset '{key}': '{gcs_path}'. Must start with 'gs://'.")
                return None

            # Check if this specific asset already exists
            if os.path.exists(local_path):
                logger.info(f"Asset '{key}' already exists at {local_path}. Skipping.")
                local_paths[key] = local_path
                continue

            logger.info(f"Downloading asset '{key}'...")
            logger.info(f"  Source: {gcs_path}")
            logger.info(f"  Destination: {local_path}")

            bucket_name = gcs_path.split('gs://', 1)[1].split('/', 1)[0]
            bucket = storage_client.bucket(bucket_name)

            allowed_files = asset_info.get('files')
            if not download_gcs_asset(bucket, gcs_path, local_path, logger, allowed_files=allowed_files):
                raise RuntimeError(f"Failed to download asset '{key}'.")

            local_paths[key] = local_path
            logger.info(f"✅ SUCCESS: Downloaded '{key}'.")

        elapsed_time = time.time() - start_time
        logger.info(f"✅ SUCCESS: All GCS assets downloaded in {elapsed_time:.2f}s.")
        return local_paths

    except exceptions.Forbidden as e:
        logger.error(
            "FAILED: GCP Permissions error. Ensure the Vertex AI job's service "
            "account has 'Storage Object Viewer' role on the relevant buckets.")
        logger.error(f"  Details: {e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return None
    except exceptions.NotFound as e:
        logger.error("FAILED: GCS bucket or path not found. Check your config.")
        logger.error(f"  Details: {e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return None
    except Exception as e:
        logger.error(f"FAILED: An unexpected error occurred during download: {e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return None
