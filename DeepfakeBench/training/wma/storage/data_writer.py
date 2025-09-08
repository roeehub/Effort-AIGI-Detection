"""
Data storage structure for backend service.

Writes received video and audio chunks to structured directories
matching the requirements from the specification.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import uuid

from wma_streaming_pb2 import ParticipantFrame, AudioBatch, ParticipantCrop


class BackendDataWriter:
    """Handles writing received data to structured directories."""
    
    def __init__(self, base_data_dir: str = "data"):
        """
        Initialize data writer.
        
        Args:
            base_data_dir: Base directory for all data storage
        """
        self.base_dir = Path(base_data_dir)
        self.video_dir = self.base_dir / "video"
        self.audio_dir = self.base_dir / "audio"
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self._ensure_directories()
        
        # Statistics
        self.stats = {
            "chunks_written": 0,
            "audio_chunks_written": 0,
            "total_participants": 0,
            "total_frames": 0,
            "total_crops": 0,
            "bytes_written": 0,
            "start_time": time.time()
        }
    
    def _ensure_directories(self) -> None:
        """Create necessary directory structure."""
        self.base_dir.mkdir(exist_ok=True)
        self.video_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
    
    def write_video_chunk(self, participant_frames: List[ParticipantFrame], 
                         uplink_metadata: Dict[str, Any]) -> str:
        """
        Write video chunk to structured directory.
        
        Args:
            participant_frames: List of ParticipantFrame messages
            uplink_metadata: Metadata from Uplink message
            
        Returns:
            Chunk directory path
        """
        if not participant_frames:
            return ""
        
        # Generate chunk ID and timestamp
        timestamp = datetime.now()
        chunk_id = f"video_chunk_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create chunk directory
        chunk_dir = self.video_dir / chunk_id
        chunk_dir.mkdir(exist_ok=True)
        
        # Write manifest for the chunk
        chunk_manifest = {
            "chunk_id": chunk_id,
            "timestamp": timestamp.isoformat(),
            "uplink_metadata": uplink_metadata,
            "participants": []
        }
        
        total_crops = 0
        total_bytes = 0
        
        # Process each participant
        for participant_frame in participant_frames:
            participant_id_raw = participant_frame.participant_id
            
            # CRITICAL FIX: Sanitize participant_id to remove JSON corruption
            # The participant_id field sometimes contains raw JSON instead of clean ID
            participant_id = self._sanitize_participant_id(participant_id_raw)
            
            self.logger.debug(f"Processing participant - Raw ID: {repr(participant_id_raw)}, Sanitized ID: {participant_id}")
            
            # Create participant directory
            participant_dir = chunk_dir / participant_id
            participant_dir.mkdir(exist_ok=True)
            
            # Write participant crops
            crop_files = []
            for i, crop in enumerate(participant_frame.crops):
                # Generate crop filename
                crop_filename = f"frame_{crop.sequence_number:06d}_crop_{i:03d}.jpg"
                crop_path = participant_dir / crop_filename
                
                # Write crop image
                with open(crop_path, 'wb') as f:
                    f.write(crop.image_data)
                
                total_bytes += len(crop.image_data)
                total_crops += 1
                
                # Record frame metadata (API expects 'frames' not 'crops')
                crop_files.append({
                    "id": f"{participant_id}-{crop.sequence_number}",  # Add missing 'id' field
                    "ts_ms": crop.timestamp_ms,  # Use 'ts_ms' not 'timestamp_ms'
                    "filename": crop_filename,   # Keep filename for multipart upload
                    "sequence_number": crop.sequence_number,
                    "bbox": {
                        "x": crop.bbox_x,
                        "y": crop.bbox_y,
                        "width": crop.bbox_width,
                        "height": crop.bbox_height
                    },
                    "confidence": crop.confidence,
                    "size_bytes": len(crop.image_data)
                })
            
            # Write participant manifest (API format with 'frames' not 'crops')
            participant_manifest = {
                "participant_id": participant_id,
                "chunk_id": participant_frame.chunk_id,
                "start_ts_ms": participant_frame.start_ts_ms,
                "end_ts_ms": participant_frame.end_ts_ms,
                "meeting_id": participant_frame.meeting_id,
                "session_id": participant_frame.session_id,
                "frame_count": participant_frame.frame_count,
                "frame_rate_hz": 30,  # Add optional but missing frame_rate_hz
                "original_frame_size": {
                    "width": participant_frame.original_frame_width,
                    "height": participant_frame.original_frame_height
                },
                "frames": crop_files  # Use 'frames' not 'crops' per API spec
            }
            
            # Write participant manifest
            participant_manifest_path = participant_dir / "manifest.json"
            with open(participant_manifest_path, 'w', encoding='utf-8') as f:
                json.dump(participant_manifest, f, indent=2)
            
            total_bytes += participant_manifest_path.stat().st_size
            
            # Add to chunk manifest
            chunk_manifest["participants"].append({
                "participant_id": participant_id,
                "crop_count": len(crop_files),
                "directory": participant_id
            })
        
        # Write chunk manifest
        chunk_manifest["summary"] = {
            "total_participants": len(participant_frames),
            "total_crops": total_crops,
            "total_bytes": total_bytes
        }
        
        chunk_manifest_path = chunk_dir / "chunk_manifest.json"
        with open(chunk_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_manifest, f, indent=2)
        
        # Update statistics
        self.stats["chunks_written"] += 1
        self.stats["total_participants"] += len(participant_frames)
        self.stats["total_frames"] += sum(pf.frame_count for pf in participant_frames)
        self.stats["total_crops"] += total_crops
        self.stats["bytes_written"] += total_bytes
        
        print(f"[DataWriter] Wrote video chunk {chunk_id}: {len(participant_frames)} participants, "
              f"{total_crops} crops, {total_bytes:,} bytes")
        
        return str(chunk_dir)
    
    def create_multipart_api_data(self, participant_frames: List[ParticipantFrame], 
                                 uplink_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create multipart/form-data structure per API specification.
        
        Args:
            participant_frames: List of ParticipantFrame messages
            uplink_metadata: Metadata from Uplink message
            
        Returns:
            Dict with 'manifest' (JSON) and 'files' (Dict[filename, bytes])
        """
        if not participant_frames:
            return {"manifest": {}, "files": {}}
        
        # Create API-compliant manifest
        api_manifest = {
            "participants": []
        }
        
        files_data = {}
        
        # Process each participant
        for participant_frame in participant_frames:
            participant_id_raw = participant_frame.participant_id
            
            # CRITICAL FIX: Sanitize participant_id to remove JSON corruption
            participant_id = self._sanitize_participant_id(participant_id_raw)
            
            self.logger.debug(f"Processing participant for API - Raw ID: {repr(participant_id_raw)}, Sanitized ID: {participant_id}")
            
            # Create participant entry
            participant_entry = {
                "participant_id": participant_id,
                "chunk_id": participant_frame.chunk_id,
                "start_ts_ms": participant_frame.start_ts_ms,
                "end_ts_ms": participant_frame.end_ts_ms,
                "meeting_id": participant_frame.meeting_id,
                "session_id": participant_frame.session_id,
                "frame_count": participant_frame.frame_count,
                "frame_rate_hz": 30,  # Add optional but missing frame_rate_hz
                "original_frame_size": {
                    "width": participant_frame.original_frame_width,
                    "height": participant_frame.original_frame_height
                },
                "frames": []  # Use 'frames' not 'crops' per API spec
            }
            
            # Process each crop/frame
            for i, crop in enumerate(participant_frame.crops):
                # Generate filename per API spec
                frame_filename = f"{participant_id}_frame_{crop.sequence_number:06d}_{i:03d}.jpg"
                
                # Add to files data for multipart upload
                files_data[frame_filename] = crop.image_data
                
                # Add frame entry to manifest
                frame_entry = {
                    "id": f"{participant_id}-{crop.sequence_number}",  # Required 'id' field
                    "ts_ms": crop.timestamp_ms,  # Use 'ts_ms' not 'timestamp_ms'
                    "filename": frame_filename   # Must match files[] part name
                }
                participant_entry["frames"].append(frame_entry)
            
            api_manifest["participants"].append(participant_entry)
        
        return {
            "manifest": api_manifest,
            "files": files_data
        }
    
    def write_audio_chunk(self, audio_batch: AudioBatch, 
                         uplink_metadata: Dict[str, Any]) -> str:
        """
        Write audio chunk to structured directory.
        
        Args:
            audio_batch: AudioBatch message
            uplink_metadata: Metadata from Uplink message
            
        Returns:
            Audio file path
        """
        # Generate filename with timestamp
        timestamp = datetime.fromtimestamp(audio_batch.start_ts_ms / 1000)
        filename = f"audio_{timestamp.strftime('%Y%m%d_%H%M%S')}_{audio_batch.chunk_id}.ogg"
        audio_path = self.audio_dir / filename
        
        # SIMPLIFIED: Just save the OGG file directly
        # Service 5 sends us the complete valid OGG file from Service 3
        ogg_data = audio_batch.ogg_data
        
        # Write audio file
        with open(audio_path, 'wb') as f:
            f.write(ogg_data)
        
        # Write metadata
        metadata = {
            "chunk_id": audio_batch.chunk_id,
            "meeting_id": audio_batch.meeting_id,
            "session_id": audio_batch.session_id,
            "start_ts_ms": audio_batch.start_ts_ms,
            "duration_ms": audio_batch.duration_ms,
            "audio_properties": {
                "sample_rate": audio_batch.sample_rate,
                "channels": audio_batch.channels,
                "bit_rate": audio_batch.bit_rate,
                "codec": audio_batch.codec,
                "container": audio_batch.container
            },
            "frame_count": audio_batch.frame_count if hasattr(audio_batch, 'frame_count') else 0,
            "file_size_bytes": len(ogg_data),
            "uplink_metadata": uplink_metadata,
            "timestamp": timestamp.isoformat()
        }
        
        metadata_path = self.audio_dir / f"{filename}.meta.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Update statistics
        self.stats["audio_chunks_written"] += 1
        self.stats["bytes_written"] += len(ogg_data) + metadata_path.stat().st_size
        
        print(f"[DataWriter] Wrote audio chunk {audio_batch.chunk_id}: "
              f"{audio_batch.frame_count if hasattr(audio_batch, 'frame_count') else 0} frames, {len(ogg_data):,} bytes, "
              f"{audio_batch.duration_ms}ms duration")
        
        return str(audio_path)
    
    def _sanitize_participant_id(self, participant_id_raw: str) -> str:
        """
        Sanitize participant_id to handle JSON corruption in gRPC messages.
        
        The participant_id field sometimes contains raw JSON fragments instead of clean IDs.
        This method extracts the actual participant ID and sanitizes it for filename use.
        
        Args:
            participant_id_raw: Raw participant_id from gRPC message
            
        Returns:
            Clean participant ID safe for filenames
        """
        try:
            # If it looks like a clean participant ID already, use it
            if len(participant_id_raw) < 50 and not any(c in participant_id_raw for c in ['"', '\n', ',', '{', '}']):
                # Still sanitize for filename safety
                return self._sanitize_filename(participant_id_raw)
            
            # Handle JSON corruption - extract participant ID from JSON fragment
            # Pattern like: 'participant_1015vf-400",\n  "fra'
            import re
            
            # Try to extract participant ID from corrupted data
            # Look for pattern: participant_[alphanumeric-_]+ before any JSON punctuation
            match = re.match(r'(participant_[a-zA-Z0-9\-_]+)', participant_id_raw)
            if match:
                clean_id = match.group(1)
                return self._sanitize_filename(clean_id)
            
            # Fallback: Take first part before any JSON punctuation and sanitize
            for delimiter in ['"', ',', '\n', '{', '}', '[', ']']:
                if delimiter in participant_id_raw:
                    clean_id = participant_id_raw.split(delimiter)[0].strip()
                    break
            else:
                clean_id = participant_id_raw
            
            return self._sanitize_filename(clean_id)
            
        except Exception as e:
            # Last resort: generate a safe ID
            import hashlib
            safe_id = f"participant_{hashlib.md5(participant_id_raw.encode()).hexdigest()[:8]}"
            self.logger.warning(f"Failed to sanitize participant_id '{participant_id_raw}': {e}. Using generated ID: {safe_id}")
            return safe_id
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string for safe filename use.
        
        Args:
            filename: Input string
            
        Returns:
            String safe for use as filename/directory name
        """
        import re
        # Remove/replace invalid characters for Windows filenames
        invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
        sanitized = re.sub(invalid_chars, '_', filename)
        
        # Trim whitespace and dots (Windows doesn't like trailing dots)
        sanitized = sanitized.strip(' .')
        
        # Limit length to reasonable size
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "participant_unknown"
            
        return sanitized

    def get_statistics(self) -> Dict[str, Any]:
        """Get data writer statistics."""
        uptime = time.time() - self.stats["start_time"]
        stats = self.stats.copy()
        stats["uptime_seconds"] = uptime
        
        if uptime > 0:
            stats["chunks_per_minute"] = (stats["chunks_written"] / uptime) * 60
            stats["bytes_per_second"] = stats["bytes_written"] / uptime
        else:
            stats["chunks_per_minute"] = 0
            stats["bytes_per_second"] = 0
            
        return stats
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up files older than specified age.
        
        Args:
            max_age_hours: Maximum file age in hours
            
        Returns:
            Number of files cleaned up
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        # Clean video chunks
        for chunk_dir in self.video_dir.iterdir():
            if chunk_dir.is_dir() and chunk_dir.stat().st_mtime < cutoff_time:
                self._remove_directory(chunk_dir)
                cleaned_count += 1
        
        # Clean audio files
        for audio_file in self.audio_dir.glob("*"):
            if audio_file.is_file() and audio_file.stat().st_mtime < cutoff_time:
                audio_file.unlink()
                cleaned_count += 1
        
        return cleaned_count
    
    def _remove_directory(self, directory: Path) -> None:
        """Recursively remove directory and all contents."""
        for item in directory.iterdir():
            if item.is_dir():
                self._remove_directory(item)
            else:
                item.unlink()
        directory.rmdir()