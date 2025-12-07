# Video Testing Guide for Four-Model Ensemble

## New Feature: Video Input Support ✨

You can now test the four-model ensemble directly with video files! The script will automatically:
1. Extract 8-32 frames uniformly from the video
2. Detect and crop faces using YOLO
3. Send frames to all 4 model servers
4. Return the final verdict

## Usage Examples

### Test with a Video File (Recommended)

```bash
# Basic video test (uses default 27 frames, YOLO face detection)
python test_four_model_fusion.py \
    --participant-name "Person Name" \
    --video /path/to/video.mp4

# Custom number of frames (8-32 range)
python test_four_model_fusion.py \
    --participant-name "John" \
    --video /Users/roeedar/Downloads/john_video.mp4 \
    --num-frames 16

# Use YOLO+Haar alignment (more accurate but slower)
python test_four_model_fusion.py \
    --participant-name "Jane" \
    --video /path/to/video.mp4 \
    --video-method yolo_haar
```

### Test with Image Directory (Old Method)

```bash
python test_four_model_fusion.py \
    --participant-name "Noyn" \
    --num-frames 11 \
    --image-dir /Users/roeedar/Downloads/Noyn_test
```

### Test with Mock Data (No Servers Needed)

```bash
python test_four_model_fusion.py --mock
```

## Parameters

### Required (pick one)
- `--video PATH`: Path to video file (MP4, AVI, MOV, etc.)
- `--image-dir PATH`: Directory containing JPEG frames
- `--mock`: Use mock predictions (no real API calls)

### Optional
- `--participant-name NAME`: Name for logging (default: "test_participant")
- `--num-frames N`: Number of frames to extract/use
  - For videos: 8-32 frames (default: 27)
  - For image-dir: Maximum frames to load
- `--video-method METHOD`: Face detection method for videos
  - `yolo`: Fast, simple square crop (default)
  - `yolo_haar`: Slower, better alignment with eye detection
- `--calibrators PATH`: Path to calibrators pickle file
- `--debug`: Enable verbose debug logging

## How Video Extraction Works

The script uses your existing `video_preprocessor.py` module:

1. **Uniform Sampling**: Extracts frames evenly spaced across the video
2. **Face Detection**: Uses YOLO to detect faces in each frame
3. **Face Cropping**: Crops and aligns faces to 224x224 pixels
4. **Quality Filtering**: Only includes frames with detectable faces
5. **Encoding**: Converts to JPEG bytes for API transmission

### Face Detection Methods

**`yolo` (Default - Recommended)**
- ✅ Fast and reliable
- ✅ Robust to various lighting conditions
- ✅ Simple square crop around face
- Use for: Most videos, real-time processing

**`yolo_haar` (Advanced)**
- ✅ Better alignment (uses eye detection)
- ✅ Handles rotated faces
- ⚠️ Slower (additional Haar Cascade step)
- ⚠️ May fail if eyes not visible
- Use for: High-quality videos, offline processing

## Real-World Examples

### Example 1: Test a Suspect Video

```bash
python test_four_model_fusion.py \
    --participant-name "Suspect Interview" \
    --video /Users/roeedar/Videos/suspect_2024_11_15.mp4 \
    --num-frames 20
```

**Expected Output:**
```
================================================================================
PROCESSING 20 frames for 'Suspect Interview'
================================================================================
[9rfa62j1] Sending 20 frames to http://34.14.89.109:8998/check_frame_batch
[1mjgo9w1] Sending 20 frames to http://34.76.180.225:8998/check_frame_batch
[dfsesrgu] Sending 20 frames to http://34.86.88.240:8998/check_frame_batch
[4vtny88m] Sending 20 frames to http://34.16.217.28:8998/check_frame_batch

[9rfa62j1] Got 20 predictions, mean=0.8234, max=0.9156
[1mjgo9w1] Got 20 predictions, mean=0.7891, max=0.8923
[dfsesrgu] Got 20 predictions, mean=0.8512, max=0.9234
[4vtny88m] Got 20 predictions, mean=0.8156, max=0.9001

Aggregating predictions per model...
[9rfa62j1] topk4: 20 frames → aggregated=0.901234 → calibrated=0.903567
[1mjgo9w1] softmax_b5: 20 frames → aggregated=0.887612 → calibrated=0.889234
[dfsesrgu] topk4: 20 frames → aggregated=0.914567 → calibrated=0.916789
[4vtny88m] topk4: 20 frames → aggregated=0.895678 → calibrated=0.897890

================================================================================
FINAL RESULT
================================================================================
  Participant: Suspect Interview
  Frames: 20
  Fusion Score: 0.999823
  Verdict: FAKE
  Confidence: High
================================================================================
```

### Example 2: Batch Test Multiple Videos

Create a simple bash script:

```bash
#!/bin/bash
# test_batch_videos.sh

for video in /Users/roeedar/Videos/test_set/*.mp4; do
    name=$(basename "$video" .mp4)
    echo "Testing: $name"
    
    python test_four_model_fusion.py \
        --participant-name "$name" \
        --video "$video" \
        --num-frames 16 \
        2>&1 | grep -A 5 "FINAL RESULT"
    
    echo "---"
done
```

### Example 3: Debug a Specific Video

```bash
# Enable debug logging to see frame-by-frame details
python test_four_model_fusion.py \
    --participant-name "Debug Test" \
    --video problem_video.mp4 \
    --debug
```

## Troubleshooting

### "Failed to extract frames from video"

**Causes:**
- No faces detected in any frame
- Video is corrupted or unreadable
- YOLO confidence threshold too high

**Solutions:**
```bash
# 1. Check if video opens in a player
vlc /path/to/video.mp4

# 2. Try different detection method
python test_four_model_fusion.py --video video.mp4 --video-method yolo_haar

# 3. Check video_preprocessor.py has the YOLO model
ls -lh /Users/roeedar/Library/Application\ Support/JetBrains/PyCharmCE2024.2/scratches/yolov8s-face.pt
```

### "Import error: video_preprocessor"

The script expects `video_preprocessor.py` in the parent directory:
```
DeepfakeBench/training/
├── video_preprocessor.py       ← Should be here
└── wma/
    └── test_four_model_fusion.py
```

### "Video processing is slow"

**Tips for faster processing:**

1. **Reduce frame count:**
   ```bash
   --num-frames 12  # Instead of 27
   ```

2. **Use simple YOLO method:**
   ```bash
   --video-method yolo  # Default, faster than yolo_haar
   ```

3. **Pre-extract frames** if testing same video multiple times:
   ```bash
   # Extract once
   ffmpeg -i video.mp4 -vf "fps=1" frames/frame_%04d.jpg
   
   # Test many times (faster)
   python test_four_model_fusion.py --image-dir frames/
   ```

## Frame Count Recommendations

| Video Length | Recommended Frames | Reason |
|--------------|-------------------|--------|
| < 5 seconds  | 8-12 frames | Short clips need fewer samples |
| 5-15 seconds | 16-20 frames | Standard interview/call length |
| 15-30 seconds | 24-27 frames | Training used ~27 frames |
| > 30 seconds | 32 frames (max) | Longer videos, more variation |

**Note:** More frames = better accuracy but slower processing

## Comparison: Video vs Image Directory

| Feature | `--video` | `--image-dir` |
|---------|-----------|---------------|
| **Setup** | Just provide video path | Need to pre-extract frames |
| **Face detection** | Automatic (YOLO) | Must be pre-cropped |
| **Frame selection** | Uniform sampling | Uses existing frames |
| **Speed** | Slower (extraction step) | Faster (no extraction) |
| **Best for** | Raw videos, one-off tests | Batch testing, repeated tests |

## Next Steps

Once you've tested several videos and are confident in the results:

1. **Export calibrators** from your training data (see `INTEGRATION_CHECKLIST.md`)
2. **Re-test with calibrators** to get accurate final scores
3. **Integrate into WMA server** using the validated code

---

**Quick Test Command:**

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection/DeepfakeBench/training/wma

python test_four_model_fusion.py \
    --participant-name "QuickTest" \
    --video /Users/roeedar/Downloads/test_video.mp4 \
    --num-frames 16
```

🚀 Happy testing!
