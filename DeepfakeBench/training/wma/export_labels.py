#!/usr/bin/env python3
"""
Export labeled data from labels.json into organized folders.

Output structure:
    exports/
        video/
            <participant_name>/
                real/
                    frame_001.jpg
                    frame_002.jpg
                fake/
                    frame_003.jpg
        audio/
            real/
                session_6_0.0-602.4.mp3
            fake/
                session_6_739.9-766.8.mp3
        manifest.json  (summary of all exports)
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
METADATA_FILE = "metadata.json"
LABELS_FILE = "selected_data/labels.json"
SESSION_AUDIO_DIR = Path("./session_audio")
EXPORT_DIR = Path("./exports")


def load_data():
    """Load metadata and labels."""
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    if not os.path.exists(LABELS_FILE):
        return metadata, {}
    
    with open(LABELS_FILE, 'r') as f:
        labels = json.load(f)
    
    return metadata, labels


def get_session_by_id(metadata, session_id):
    """Find session by ID."""
    for s in metadata:
        if s['id'] == session_id:
            return s
    return None


def export_video_frames(session, label, export_dir):
    """Export video frames for a label."""
    track = label['track']
    label_type = label['type']
    start_time = label['start']
    end_time = label['end']
    
    # Skip audio track
    if track == 'audio':
        return []
    
    # Get participant frames
    participants = session.get('participants', {})
    if track not in participants:
        print(f"    Warning: Participant '{track}' not found in session {session['id']}")
        return []
    
    frames = participants[track]
    
    # Create output directory
    output_dir = export_dir / "video" / track / label_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find frames within time range
    exported = []
    for frame in frames:
        frame_time = frame['t0']
        if start_time <= frame_time <= end_time:
            src_path = frame['src']
            if os.path.exists(src_path):
                # Create unique filename with timestamp
                filename = f"s{session['id']}_{frame_time:.1f}_{os.path.basename(src_path)}"
                dst_path = output_dir / filename
                
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                
                exported.append({
                    "src": str(dst_path),
                    "session_id": session['id'],
                    "participant": track,
                    "type": label_type,
                    "time": frame_time
                })
    
    return exported


def export_audio_segment(session, label, export_dir):
    """Export audio segment for a label."""
    track = label['track']
    
    # Only process audio track labels
    if track != 'audio':
        return None
    
    label_type = label['type']
    start_time = label['start']
    end_time = label['end']
    session_id = session['id']
    
    # Find session audio file
    session_audio = SESSION_AUDIO_DIR / f"session_{session_id}.mp3"
    if not session_audio.exists():
        print(f"    Warning: Session audio not found: {session_audio}")
        return None
    
    # Create output directory
    output_dir = export_dir / "audio" / label_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename
    duration = end_time - start_time
    filename = f"session_{session_id}_{start_time:.1f}-{end_time:.1f}.mp3"
    output_path = output_dir / filename
    
    if output_path.exists():
        # Already exported
        return {
            "src": str(output_path),
            "session_id": session_id,
            "type": label_type,
            "start": start_time,
            "end": end_time,
            "duration": duration
        }
    
    # Use ffmpeg to extract segment
    cmd = [
        'ffmpeg', '-y',
        '-i', str(session_audio),
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:a', 'libmp3lame',
        '-b:a', '128k',
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"    FFmpeg error: {result.stderr[-200:]}")
            return None
        
        return {
            "src": str(output_path),
            "session_id": session_id,
            "type": label_type,
            "start": start_time,
            "end": end_time,
            "duration": duration
        }
    except Exception as e:
        print(f"    Error extracting audio: {e}")
        return None


def process_exports():
    """Main export function. Returns manifest of exported files."""
    metadata, labels = load_data()
    
    if not labels:
        return {"error": "No labels found", "video": [], "audio": []}
    
    # Create export directory
    EXPORT_DIR.mkdir(exist_ok=True)
    
    manifest = {
        "exported_at": datetime.now().isoformat(),
        "video": [],
        "audio": [],
        "stats": {
            "video_real": 0,
            "video_fake": 0,
            "audio_real": 0,
            "audio_fake": 0
        }
    }
    
    # Process each session's labels
    for session_id_str, session_labels in labels.items():
        session_id = int(session_id_str)
        session = get_session_by_id(metadata, session_id)
        
        if not session:
            print(f"Warning: Session {session_id} not found in metadata")
            continue
        
        print(f"Processing session {session_id} ({len(session_labels)} labels)...")
        
        for label in session_labels:
            track = label['track']
            label_type = label['type']
            
            if track == 'audio':
                # Export audio segment
                result = export_audio_segment(session, label, EXPORT_DIR)
                if result:
                    manifest['audio'].append(result)
                    manifest['stats'][f'audio_{label_type}'] += 1
                    print(f"  Audio [{label_type}]: {label['start']:.1f}s - {label['end']:.1f}s")
            else:
                # Export video frames
                frames = export_video_frames(session, label, EXPORT_DIR)
                manifest['video'].extend(frames)
                manifest['stats'][f'video_{label_type}'] += len(frames)
                print(f"  Video [{label_type}] {track}: {len(frames)} frames")
    
    # Save manifest
    manifest_path = EXPORT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nExport complete!")
    print(f"  Video frames: {manifest['stats']['video_real']} real, {manifest['stats']['video_fake']} fake")
    print(f"  Audio clips: {manifest['stats']['audio_real']} real, {manifest['stats']['audio_fake']} fake")
    print(f"  Manifest saved to: {manifest_path}")
    
    return manifest


if __name__ == "__main__":
    process_exports()
