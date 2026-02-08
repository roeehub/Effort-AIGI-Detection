#!/usr/bin/env python3
"""
Concatenate audio chunks into single session audio files.
Creates one continuous audio file per session with silence filling gaps.

Uses a two-pass approach for scalability:
1. Generate silence segments to fill gaps
2. Concatenate all segments (audio + silence) in order using ffmpeg concat

Usage:
    python3 concatenate_audio.py           # Skip existing files
    python3 concatenate_audio.py --force   # Regenerate all files
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

# Configuration
METADATA_FILE = "metadata.json"
OUTPUT_DIR = Path("./session_audio")
TEMP_DIR = Path("./session_audio/temp")

# Parse --force flag
FORCE_REGENERATE = "--force" in sys.argv or "-f" in sys.argv


def create_silence_file(duration_sec, output_path):
    """Create a silent audio file of specified duration."""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'anullsrc=r=44100:cl=stereo',
        '-t', str(duration_sec),
        '-c:a', 'libmp3lame',
        '-b:a', '128k',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.returncode == 0


def create_session_audio(session):
    """Create a single audio file for a session using concat approach."""
    session_id = session['id']
    duration = session['duration']
    audio_chunks = session.get('audio', [])
    
    if not audio_chunks:
        print(f"  Session {session_id}: No audio chunks, skipping")
        return None
    
    output_file = OUTPUT_DIR / f"session_{session_id}.mp3"
    
    # If already exists, skip (unless --force)
    if output_file.exists() and not FORCE_REGENERATE:
        print(f"  Session {session_id}: Already exists, skipping (use --force to regenerate)")
        return str(output_file)
    
    print(f"  Session {session_id}: {len(audio_chunks)} chunks, {duration:.1f}s duration")
    
    # Filter valid chunks and sort by time
    valid_chunks = [c for c in audio_chunks if os.path.exists(c['src'])]
    if not valid_chunks:
        print(f"    No valid audio files found, skipping")
        return None
    
    valid_chunks.sort(key=lambda x: x['t0'])
    
    # Create temp directory for this session
    session_temp = TEMP_DIR / f"session_{session_id}"
    session_temp.mkdir(parents=True, exist_ok=True)
    
    # Build list of segments (silence gaps + audio chunks)
    segments = []
    current_time = 0.0
    
    for i, chunk in enumerate(valid_chunks):
        chunk_start = chunk['t0']
        chunk_duration = chunk.get('t1', chunk_start + 4.0) - chunk_start  # Default 4s
        
        # Add silence gap if needed
        gap = chunk_start - current_time
        if gap > 0.1:  # Only add if gap > 100ms
            silence_file = session_temp / f"silence_{i}.mp3"
            if not silence_file.exists():
                if not create_silence_file(gap, str(silence_file)):
                    print(f"    Warning: Failed to create silence for gap at {current_time:.1f}s")
                    continue
            segments.append(str(silence_file))
        
        # Add the audio chunk (convert to consistent format if needed)
        segments.append(chunk['src'])
        current_time = chunk_start + chunk_duration
    
    # Add trailing silence if needed
    if current_time < duration - 0.1:
        trailing_gap = duration - current_time
        silence_file = session_temp / "silence_end.mp3"
        if create_silence_file(trailing_gap, str(silence_file)):
            segments.append(str(silence_file))
    
    if not segments:
        print(f"    No segments to concatenate")
        return None
    
    # Create concat list file with ABSOLUTE paths
    concat_file = session_temp / "concat_list.txt"
    with open(concat_file, 'w') as f:
        for seg in segments:
            # Use absolute path to avoid path resolution issues
            abs_path = os.path.abspath(seg)
            # Escape single quotes in path
            escaped_path = abs_path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    
    # Run ffmpeg concat
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c:a', 'libmp3lame',
        '-b:a', '128k',
        '-ac', '2',
        '-ar', '44100',
        str(output_file)
    ]
    
    try:
        print(f"    Concatenating {len(segments)} segments...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            # Get last 500 chars of stderr which usually has the actual error
            error_msg = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
            print(f"    FFmpeg error (code {result.returncode}): {error_msg}")
            return None
        print(f"    Created: {output_file}")
        
        # Clean up temp files for this session
        import shutil
        shutil.rmtree(session_temp, ignore_errors=True)
        
        return str(output_file)
    except subprocess.TimeoutExpired:
        print(f"    Timeout creating session {session_id}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    # Load metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found. Run indexer.py first.")
        return
    
    with open(METADATA_FILE, 'r') as f:
        sessions = json.load(f)
    
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    
    print(f"Processing {len(sessions)} sessions...")
    
    # Process each session
    results = {}
    for session in sessions:
        audio_path = create_session_audio(session)
        if audio_path:
            results[session['id']] = audio_path
    
    # Save mapping file
    mapping_file = OUTPUT_DIR / "session_audio_map.json"
    with open(mapping_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    print(f"\nDone! Created {len(results)} session audio files.")
    print(f"Mapping saved to {mapping_file}")


if __name__ == "__main__":
    main()
