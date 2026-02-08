import os
import json
import re
import datetime
from pathlib import Path

# ================= CONFIGURATION =================
BASE_DIR = Path("./data")
AUDIO_DIR = BASE_DIR / "audio"
VIDEO_DIR = BASE_DIR / "video"
OUTPUT_FILE = "metadata.json"

# Gap in seconds to consider a new session (e.g., 5 minutes)
SESSION_GAP_THRESHOLD = 300 
# =================================================

def parse_timestamp_from_filename(filename):
    """Extracts unix timestamp (ms) from audio_1762...mp3"""
    match = re.search(r'audio_(\d+)_', filename)
    if match:
        return int(match.group(1)) / 1000.0
    return None

def parse_timestamp_from_video_chunk(dirname):
    """Extracts timestamp from video_chunk_20251103_120749_..."""
    # Format: YYYYMMDD_HHMMSS
    match = re.search(r'video_chunk_(\d{8})_(\d{6})_', dirname)
    if match:
        dt_str = f"{match.group(1)}_{match.group(2)}"
        dt = datetime.datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
        return dt.timestamp()
    return None

def scan_audio():
    print("Scanning audio...")
    audio_files = []
    if not AUDIO_DIR.exists():
        print(f"Warning: {AUDIO_DIR} does not exist.")
        return []
        
    for f in os.listdir(AUDIO_DIR):
        if f.endswith('.mp3'):
            ts = parse_timestamp_from_filename(f)
            if ts:
                # Assuming 4 seconds duration as stated
                audio_files.append({
                    "type": "audio",
                    "path": str(AUDIO_DIR / f),
                    "start": ts,
                    "end": ts + 4.0,
                    "name": f
                })
    return audio_files

def scan_video():
    print("Scanning video...")
    video_chunks = []
    if not VIDEO_DIR.exists():
        return []

    # Sort chunks to help with processing, though not strictly necessary
    chunk_dirs = sorted([d for d in os.listdir(VIDEO_DIR) if d.startswith("video_chunk_")])

    for chunk in chunk_dirs:
        chunk_ts = parse_timestamp_from_video_chunk(chunk)
        if not chunk_ts:
            continue
            
        chunk_path = VIDEO_DIR / chunk
        
        # Iterate participants inside the chunk
        for participant in os.listdir(chunk_path):
            # Filter participants
            if participant.startswith("participant") or participant.startswith("."):
                continue
                
            part_path = chunk_path / participant
            if not part_path.is_dir():
                continue

            # Find a representative frame (first .jpg found)
            frames = sorted([f for f in os.listdir(part_path) if f.endswith(".jpg")])
            if frames:
                video_chunks.append({
                    "type": "video",
                    "participant": participant,
                    "start": chunk_ts,
                    "end": chunk_ts + 1.0, # Assume chunk implies existence at this second
                    "path": str(part_path / frames[0]), # Path to image
                    "chunk_name": chunk
                })
                
    return video_chunks

def build_sessions(items):
    print("Building sessions...")
    if not items:
        return []

    # Sort everything by start time
    items.sort(key=lambda x: x['start'])

    sessions = []
    current_session = {"id": 1, "start": items[0]['start'], "end": items[0]['end'], "items": []}
    
    for item in items:
        # Check gap
        if item['start'] - current_session['end'] > SESSION_GAP_THRESHOLD:
            # Close session
            sessions.append(current_session)
            # Start new
            current_session = {
                "id": len(sessions) + 1, 
                "start": item['start'], 
                "end": item['end'], 
                "items": []
            }
        
        # Add item
        current_session['items'].append(item)
        # Extend session end if needed
        if item['end'] > current_session['end']:
            current_session['end'] = item['end']
            
    sessions.append(current_session)
    return sessions

def main():
    audio_data = scan_audio()
    video_data = scan_video()
    
    all_data = audio_data + video_data
    print(f"Found {len(audio_data)} audio clips and {len(video_data)} video chunks.")
    
    sessions = build_sessions(all_data)
    print(f"Grouped into {len(sessions)} sessions.")
    
    # Organize data for frontend: 
    # { session_id: { start, end, audio: [], participants: { name: [] } } }
    final_output = []
    
    for s in sessions:
        s_obj = {
            "id": s['id'],
            "start_time": s['start'],
            "end_time": s['end'],
            "duration": s['end'] - s['start'],
            "audio": [],
            "participants": {}
        }
        
        for item in s['items']:
            rel_start = item['start'] - s['start']
            rel_end = item['end'] - s['start']
            
            if item['type'] == 'audio':
                s_obj['audio'].append({
                    "src": item['path'],
                    "t0": rel_start,
                    "t1": rel_end,
                    "abs_t": item['start']
                })
            elif item['type'] == 'video':
                p_name = item['participant']
                if p_name not in s_obj['participants']:
                    s_obj['participants'][p_name] = []
                s_obj['participants'][p_name].append({
                    "src": item['path'],
                    "t0": rel_start,
                    "abs_t": item['start']
                })
        
        final_output.append(s_obj)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()