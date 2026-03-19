"""
Reporting mixin for Trainer.

Provides functionality for generating and uploading validation reports
including CSV files and summary reports to GCS.
"""
import os
import csv
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


class ReportingMixin:
    """
    Mixin that provides report generation and upload functionality.
    
    Assumes the base class has:
    - self.config: Training configuration dict
    - self.logger: Logger instance
    - self._upload_to_gcs(): Method to upload files to GCS
    """
    
    def generate_frame_report(
        self,
        frame_data: List[List[Any]],
        output_path: str
    ) -> bool:
        """
        Generate a frame-level CSV report.
        
        Args:
            frame_data: List of rows [method, label, video_id, frame_path, frame_prob]
            output_path: Path to write the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['method', 'label', 'video_id', 'frame_path', 'frame_prob'])
                writer.writerows(frame_data)
            self.logger.info(f"Frame report generated with {len(frame_data)} entries.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate frame report: {e}")
            return False
    
    def generate_video_report(
        self,
        video_data: List[List[Any]],
        output_path: str
    ) -> bool:
        """
        Generate a video-level CSV report.
        
        Args:
            video_data: List of rows [method, label, video_id, avg_prob, prediction, is_correct]
            output_path: Path to write the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['method', 'label', 'video_id', 'avg_video_prob', 'prediction', 'is_correct'])
                writer.writerows(video_data)
            self.logger.info(f"Video report generated with {len(video_data)} entries.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate video report: {e}")
            return False
    
    def generate_summary_report(
        self,
        log_prefix: str,
        run_name: str,
        video_data: List[List[Any]],
        output_path: str
    ) -> bool:
        """
        Generate a text summary report.
        
        Args:
            log_prefix: Validation log prefix
            run_name: Name of the run
            video_data: Video-level data for calculations
            output_path: Path to write the report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import here to avoid runtime import at module load
            import numpy as np
            from sklearn.metrics import confusion_matrix
            from collections import defaultdict
            
            with open(output_path, 'w') as f:
                f.write(f"Validation Summary Report for: {log_prefix}\n")
                f.write(f"Run Name: {run_name}\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 40 + "\n")
                f.write("Overall Performance\n")
                f.write("-" * 40 + "\n")
                
                if video_data:
                    video_labels = [row[1] for row in video_data]
                    video_preds = [row[4] for row in video_data]
                    
                    tn, fp, fn, tp = confusion_matrix(
                        video_labels, video_preds, labels=[0, 1]
                    ).ravel()
                    total = len(video_labels)
                    acc = (tp + tn) / total if total > 0 else 0
                    
                    f.write(f"Total Videos: {total}\n")
                    f.write(f"Accuracy: {acc:.4f}\n")
                    f.write(f"True Positives (Correctly identified Fake): {tp}\n")
                    f.write(f"True Negatives (Correctly identified Real): {tn}\n")
                    f.write(f"False Positives (Real misclassified as Fake): {fp}\n")
                    f.write(f"False Negatives (Fake misclassified as Real): {fn}\n\n")
                    
                    # Per-method breakdown
                    f.write("=" * 40 + "\n")
                    f.write("Per-Method Performance\n")
                    f.write("-" * 40 + "\n")
                    
                    method_video_data = defaultdict(list)
                    for row in video_data:
                        method_video_data[row[0]].append(row)
                    
                    for method in sorted(method_video_data.keys()):
                        method_rows = method_video_data[method]
                        if not method_rows:
                            continue
                        
                        labels = np.array([row[1] for row in method_rows])
                        predictions = np.array([row[4] for row in method_rows])
                        
                        is_real_method = (labels[0] == 0)
                        method_type = "REAL" if is_real_method else "FAKE"
                        
                        f.write(f"----- Method: {method} ({method_type}) -----\n")
                        f.write(f"Total Videos: {len(labels)}\n")
                        
                        correct_predictions = np.sum(labels == predictions)
                        accuracy = correct_predictions / len(labels)
                        f.write(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{len(labels)} correct)\n\n")
            
            self.logger.info("Summary report generated.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            return False
    
    def deduplicate_frame_data(
        self,
        frame_data: List[List[Any]]
    ) -> List[List[Any]]:
        """
        Deduplicate frame data by method+frame_path.
        
        Args:
            frame_data: Raw frame data with potential duplicates
            
        Returns:
            Deduplicated frame data
        """
        seen_keys = set()
        deduplicated = []
        
        for row in frame_data:
            method = row[0]
            frame_path = row[3]
            unique_key = f"{method}_{frame_path}"
            
            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                deduplicated.append(row)
        
        original_count = len(frame_data)
        new_count = len(deduplicated)
        
        if original_count != new_count:
            self.logger.info(
                f"Deduplication: Frame data reduced from {original_count} to {new_count} entries"
            )
        
        return deduplicated
    
    def deduplicate_video_data(
        self,
        video_data: List[List[Any]]
    ) -> List[List[Any]]:
        """
        Deduplicate video data by method+video_id.
        
        Args:
            video_data: Raw video data with potential duplicates
            
        Returns:
            Deduplicated video data
        """
        seen_keys = set()
        deduplicated = []
        
        for row in video_data:
            method = row[0]
            video_id = row[2]
            unique_key = f"{method}_{video_id}"
            
            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                deduplicated.append(row)
        
        original_count = len(video_data)
        new_count = len(deduplicated)
        
        if original_count != new_count:
            self.logger.info(
                f"Deduplication: Video data reduced from {original_count} to {new_count} entries"
            )
        
        return deduplicated
    
    def generate_and_upload_reports(
        self,
        log_prefix: str,
        frame_data: List[List[Any]],
        video_data: List[List[Any]],
        run_name: str = "",
        gcs_base_path: str = "gs://training-job-outputs/test_results"
    ) -> Tuple[int, int]:
        """
        Generate all reports and upload to GCS.
        
        Args:
            log_prefix: Validation log prefix
            frame_data: Frame-level data (already deduplicated if needed)
            video_data: Video-level data (already deduplicated if needed)
            run_name: Name of the run for the summary
            gcs_base_path: GCS bucket path for uploads
            
        Returns:
            Tuple of (files_generated, files_uploaded)
        """
        local_temp_dir = tempfile.mkdtemp()
        files_generated = []
        files_uploaded = 0
        
        try:
            # 1. Frame report
            frame_csv_path = os.path.join(local_temp_dir, 'frames_report.csv')
            if self.generate_frame_report(frame_data, frame_csv_path):
                files_generated.append(('frames_report.csv', frame_csv_path))
            
            # 2. Video report
            video_csv_path = os.path.join(local_temp_dir, 'videos_report.csv')
            if self.generate_video_report(video_data, video_csv_path):
                files_generated.append(('videos_report.csv', video_csv_path))
            
            # 3. Summary report
            summary_txt_path = os.path.join(local_temp_dir, 'summary_report.txt')
            if self.generate_summary_report(log_prefix, run_name, video_data, summary_txt_path):
                files_generated.append(('summary_report.txt', summary_txt_path))
            
            # 4. Upload to GCS
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            gcs_folder = f"{timestamp}_{log_prefix}"
            
            for filename, local_path in files_generated:
                try:
                    gcs_path = f"{gcs_base_path}/{gcs_folder}/{filename}"
                    self._upload_to_gcs(local_path, gcs_path)
                    files_uploaded += 1
                except Exception as e:
                    self.logger.error(f"Failed to upload {filename} to GCS: {e}")
            
            if files_generated:
                self.logger.info(
                    f"Successfully processed {files_uploaded}/{len(files_generated)} report files."
                )
            else:
                self.logger.warning("No report files were successfully generated.")
            
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(local_temp_dir)
            except Exception as e:
                self.logger.error(f"Failed to clean up temporary directory: {e}")
        
        return len(files_generated), files_uploaded
    
    def collect_sanity_check_data(
        self,
        method: str,
        video_id: str,
        frame_probs: List[float],
        labels: List[int],
        frame_paths: List[str],
        epoch: int,
        step_cnt: int,
        max_frames: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Collect sanity check data for model consistency verification.
        
        Args:
            method: Method name
            video_id: Video identifier
            frame_probs: List of per-frame probabilities
            labels: List of labels
            frame_paths: List of frame paths
            epoch: Current epoch
            step_cnt: Current step
            max_frames: Maximum frames to collect
            
        Returns:
            List of sanity check data dictionaries
        """
        sanity_data = []
        
        for frame_idx in range(min(max_frames, len(frame_probs))):
            frame_path = (
                frame_paths[frame_idx]
                if frame_idx < len(frame_paths)
                else f"video_{video_id}_frame_{frame_idx}"
            )
            label = labels[0] if labels else -1
            
            sanity_data.append({
                'method': method,
                'video_id': video_id,
                'frame_idx': frame_idx,
                'frame_path': frame_path,
                'probability': float(frame_probs[frame_idx]),
                'label': int(label),
                'epoch': epoch + 1,
                'step': step_cnt
            })
        
        return sanity_data
