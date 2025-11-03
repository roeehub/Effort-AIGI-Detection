# Hysteresis / Flagged Mode Implementation

## Overview

Implemented a peak-based hysteresis system to stabilize participant banner states and prevent rapid oscillation after high-suspicion flags.

## Problem Solved

Previously, a participant with probability hovering around 0.75-0.80 could rapidly bounce between YELLOW and RED banners, causing visual flicker and reducing user confidence in the system.

## Solution: Two-Mode Threshold System

### Normal Mode (Default)
- **GREEN**: mean_prob < (threshold - margin) = 0.70
- **YELLOW**: 0.70 ≤ mean_prob < 0.80
- **RED**: mean_prob ≥ 0.80

### Flagged Mode (Activated when peak ≥ 0.90)
- **GREEN**: mean_prob < 0.50 (requires significant drop)
- **YELLOW**: 0.50 ≤ mean_prob < 0.70 (wider band)
- **RED**: mean_prob ≥ 0.70 (much lower threshold than normal)

## How It Works

### 1. Peak Tracking
Every time a participant's mean probability is calculated, we update their peak:
```python
if mean_prob > state.peak_prob:
    state.peak_prob = mean_prob
```

### 2. Flagged Mode Activation
When peak crosses **0.90**, flagged mode activates:
```python
if not state.is_flagged and state.peak_prob >= FLAGGED_TRIGGER:
    state.is_flagged = True
    # Log: "⚠️ FLAGGED MODE ACTIVATED"
```

### 3. Strict Recovery Thresholds
While flagged, the system uses stricter thresholds that make it harder to recover:
- Someone at 0.92 (RED) who drops to 0.75 stays RED (not YELLOW)
- Someone at 0.75 (RED) who drops to 0.65 goes to YELLOW (not GREEN)
- They must drop below 0.50 to reach GREEN

### 4. Mode Reset
Once GREEN is achieved in flagged mode, the system resets:
```python
if state.is_flagged and new_verdict == pb2.GREEN:
    state.is_flagged = False
    state.peak_prob = mean_prob  # Reset peak
    # Log: "✓ FLAGGED MODE RESET"
```

## Configuration Constants

All thresholds are configurable at the top of `participant_manager.py`:

```python
# Flagged mode trigger
FLAGGED_TRIGGER = 0.90  # Peak that activates strict mode

# Flagged mode thresholds
FLAGGED_GREEN_UPPER = 0.50    # Must drop below 0.50 for GREEN
FLAGGED_YELLOW_LOWER = 0.50   # YELLOW starts at 0.50
FLAGGED_YELLOW_UPPER = 0.70   # YELLOW ends at 0.70
FLAGGED_RED_LOWER = 0.70      # RED if >= 0.70
```

## Example Scenario

### Normal Participant (Never Flagged)
| Batch | Mean Prob | Mode   | Verdict | Banner Sent |
|-------|-----------|--------|---------|-------------|
| 1     | 0.65      | NORMAL | GREEN   | ✓ (new)     |
| 2     | 0.73      | NORMAL | YELLOW  | ✓ (change)  |
| 3     | 0.68      | NORMAL | GREEN   | ✓ (change)  |

### High-Suspicion Participant (Gets Flagged)
| Batch | Mean Prob | Peak  | Mode    | Verdict | Banner Sent | Notes |
|-------|-----------|-------|---------|---------|-------------|-------|
| 1     | 0.65      | 0.65  | NORMAL  | GREEN   | ✓ (new)     | -     |
| 2     | 0.82      | 0.82  | NORMAL  | RED     | ✓ (change)  | -     |
| 3     | 0.93      | **0.93** | **FLAGGED** | RED     | ✗ (same)    | **⚠️ FLAGGED MODE ACTIVATED!** |
| 4     | 0.75      | 0.93  | FLAGGED | RED     | ✗ (same)    | Stays RED (≥0.70 in flagged) |
| 5     | 0.65      | 0.93  | FLAGGED | YELLOW  | ✓ (change)  | 0.50-0.70 = YELLOW in flagged |
| 6     | 0.60      | 0.93  | FLAGGED | YELLOW  | ✗ (same)    | Still in yellow band |
| 7     | 0.45      | 0.93  | FLAGGED | GREEN   | ✓ (change)  | **✓ FLAGGED MODE RESET!** |
| 8     | 0.72      | 0.72  | NORMAL  | YELLOW  | ✓ (change)  | Back to normal thresholds |

## Log Messages

### Flagged Mode Activation
```
[ParticipantManager] ⚠️  FLAGGED MODE ACTIVATED for participant_123! 
Peak probability 0.930 >= 0.900. 
Now using strict recovery thresholds: GREEN<0.50, YELLOW=0.50-0.70, RED>=0.70
```

### Regular State Logging (Enhanced)
```
[ParticipantManager] ID: participant_123, MeanProb: 0.650, Peak: 0.930, Mode: FLAGGED, 
NewVerdict: YELLOW, OldVerdict: RED, Changed: True, Counter: 1/1, IsNew: False
```

### Flagged Mode Reset
```
[ParticipantManager] ✓ FLAGGED MODE RESET for participant_123! 
Reached GREEN (mean_prob=0.450 < 0.50). Peak was 0.930. Returning to normal thresholds.
```

### Banner Trigger with Mode Indicator
```
[ParticipantManager] TRIGGER! Sending verdict for participant_123. Reason: Change. [FLAGGED MODE]
```

## Implementation Details

### Modified Files
- `participant_manager.py`: Core logic for hysteresis

### Key Changes
1. **Added constants** (lines 29-43): Configuration for flagged mode
2. **Updated ParticipantState** (lines 57-59): Added `peak_prob` and `is_flagged` fields
3. **Updated _calculate_band_level** (lines 87-113): Dual-mode threshold calculation
4. **Enhanced process_and_decide** (lines 176-223): Peak tracking, flagging, and reset logic

### Data Structure Changes
```python
@dataclass
class ParticipantState:
    # ... existing fields ...
    peak_prob: float = DEFAULT_START_PROB  # NEW: Track highest mean probability
    is_flagged: bool = False               # NEW: Track if in flagged mode
```

## Benefits

1. **Prevents Flicker**: High-suspicion participants don't rapidly oscillate between states
2. **Confidence-Based**: Only activates for truly suspicious behavior (≥0.90)
3. **Eventual Recovery**: Participants can still reach GREEN, but must prove sustained improvement
4. **Clear Logging**: Activation and reset are logged with warning-level visibility
5. **Configurable**: All thresholds can be tuned without code changes
6. **Simple**: Peak-based approach is easy to understand and debug

## Testing Recommendations

1. **Normal Operation**: Verify participants with prob < 0.90 never enter flagged mode
2. **Flagged Activation**: Test that peak ≥ 0.90 triggers flagged mode
3. **Sticky Recovery**: Verify participant at 0.92 who drops to 0.75 stays RED
4. **Full Recovery**: Verify participant can reach GREEN at < 0.50 and mode resets
5. **Log Visibility**: Check that activation/reset messages appear in logs

## Future Enhancements (Optional)

1. **Timeout-based reset**: Auto-reset flagged mode after N minutes of stability
2. **Graduated recovery**: Multiple flagged tiers (0.90, 0.95, 0.99) with different thresholds
3. **Historical peak decay**: Gradually reduce peak_prob over time if stable
4. **Per-participant tuning**: Different thresholds based on participant history
