# WMA Backend Live Status Dashboard - Implementation Guide

**Last Updated:** November 2, 2025  
**For:** Development Team  
**Target:** Real-time monitoring of participant frame processing with minimal overhead

---

## 📋 Executive Summary

We need to implement a **live status dashboard** that provides real-time visibility into participant frame processing. The dashboard will display continuously updating information about active participants, their detection status, queue sizes, and system health metrics.

**Key Requirements:**
- ✅ Update every **2 seconds** (configurable)
- ✅ **TOP-style** display (overwrites itself) for live viewing via SSH
- ✅ **Color-coded** output using `rich` library
- ✅ **ASCII sparklines** for probability trends
- ✅ **Separate history file** that persists across updates but resets on server restart
- ✅ **Focus on video/participants only** (no audio monitoring needed)
- ✅ Minimal performance overhead

---

## 🎯 Goals & Benefits

### What This Solves:
1. **Real-time visibility** - See what's happening without parsing logs
2. **Quick diagnostics** - Identify stuck participants or processing issues
3. **Performance monitoring** - Track queue sizes and API response times
4. **Trend analysis** - See if probabilities are increasing/decreasing over time

### User Workflow:
```bash
# Terminal 1: Run the server
python start_backend.py --dashboard-enabled

# Terminal 2: Watch the live dashboard
tail -f wma_status_dashboard.log
```

---

## 🏗️ Architecture Overview

### Components to Build:

```
┌─────────────────────────────────────────────────────┐
│ server.py (existing)                                │
│ ├── StreamingServiceImpl                           │
│ │   ├── ParticipantManager (existing)              │
│ │   ├── VideoIOWorker (existing)                   │
│ │   └── VideoAPIManager (existing)                 │
│ └── NEW: StatusDashboardMonitor ──────────────────┐ │
└────────────────────────────────────────────────────┼─┘
                                                     │
                    ┌────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────┐
        │ StatusDashboardMonitor        │
        │ (new class in server.py)      │
        ├───────────────────────────────┤
        │ - Background thread           │
        │ - Polls every 2 seconds       │
        │ - Collects state data         │
        │ - Formats output              │
        │ - Writes to files             │
        └───────────────────────────────┘
                    │
                    ├─────────────────┬────────────────┐
                    ▼                 ▼                ▼
        wma_status_dashboard.log  wma_dashboard_  (Console)
        (live, overwrites)        history.log     (optional)
                                  (append-only)
```

---

## 📁 File Structure

```
wma/
├── server.py                          # MODIFY: Add StatusDashboardMonitor class
├── participant_manager.py             # MODIFY: Add state export methods
├── start_backend.py                   # MODIFY: Add CLI flags
├── requirements.txt                   # MODIFY: Add 'rich' dependency
├── wma_server.log                     # EXISTING: Detailed logs
├── wma_status_dashboard.log           # NEW: Live dashboard (overwritten every 2s)
└── wma_dashboard_history.log          # NEW: Timestamped snapshots (append-only)
```

---

## 🔧 Implementation Tasks

---

## **TASK 1: Add Dependencies**

**File:** `requirements.txt`

**What to add:**
```
rich>=13.0.0
```

**Why:** The `rich` library provides:
- Beautiful terminal formatting
- Color support over SSH
- Easy table creation
- Progress bars and status indicators

**Installation:**
```bash
pip install rich
```

---

## **TASK 2: Enhance ParticipantManager with State Export**

**File:** `participant_manager.py`

**Location:** Add new methods to the `ParticipantManager` class (around line 100, after existing methods)

### What to add:

```python
def get_all_participant_summaries(self) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed summaries for all participants.
    Thread-safe method to export current state.
    
    Returns:
        Dict mapping participant_id to summary dict containing:
        - history: List of recent probabilities
        - current_verdict: Current banner level
        - mean_prob: Current mean probability
        - batch_counter: Batches since last verdict change
        - last_seen_ts: Last update timestamp
        - is_active: Whether participant is currently active
        - frames_processed: Total frames processed (estimate)
    """
    with self.lock:
        summaries = {}
        now = time.time()
        
        for pid, state in self.participants.items():
            # Calculate activity status
            inactive_seconds = now - state.last_seen_ts
            is_active = inactive_seconds < 10.0  # Active if seen in last 10s
            
            # Get recent history (last 10 values for sparkline)
            recent_history = list(state.history)[:10]
            
            # Calculate mean from active window
            active_window_probs = [
                state.history[i] 
                for i in range(min(len(state.history), ACTIVE_TEST_WINDOW))
            ]
            mean_prob = float(np.mean(active_window_probs)) if active_window_probs else DEFAULT_START_PROB
            
            summaries[pid] = {
                'history': recent_history,
                'full_history_size': len(state.history),
                'current_verdict': state.current_verdict,
                'mean_prob': mean_prob,
                'batch_counter': state.batch_counter,
                'last_seen_ts': state.last_seen_ts,
                'inactive_seconds': inactive_seconds,
                'is_active': is_active,
                'frames_processed': HISTORY_WINDOW_SIZE - state.history.count(DEFAULT_START_PROB),
                'is_new': state.is_new
            }
        
        return summaries

def get_manager_stats(self) -> Dict[str, Any]:
    """
    Get high-level statistics about the manager.
    
    Returns:
        Dict with manager-level stats
    """
    with self.lock:
        return {
            'total_participants': len(self.participants),
            'active_participants': sum(
                1 for state in self.participants.values() 
                if (time.time() - state.last_seen_ts) < 10.0
            ),
            'threshold': self.threshold,
            'margin': self.margin,
            'history_window_size': HISTORY_WINDOW_SIZE,
            'active_test_window': ACTIVE_TEST_WINDOW
        }
```

### Why:
- **Thread-safe access** to participant state without exposing internal locks
- **Encapsulation** - keeps dashboard logic separate from core business logic
- **Performance** - Minimizes time holding locks by batching data collection
- **Flexibility** - Returns structured data that can be formatted multiple ways

---

## **TASK 3: Create StatusDashboardMonitor Class**

**File:** `server.py`

**Location:** Add this class **after** the `VideoAPIManager` class and **before** the `StreamingServiceImpl` class (around line 450)

### Full Class Implementation:

```python
# ---------- Status Dashboard Monitor ----------
class StatusDashboardMonitor:
    """
    Background monitor that generates live status dashboard updates.
    
    Runs in a separate thread and periodically writes formatted status
    to dashboard files for live monitoring via tail -f.
    """
    
    def __init__(
        self, 
        participant_manager: 'ParticipantManager',
        video_io_workers: List['MediaIOWorker'],
        video_api: 'VideoAPIManager',
        update_interval: float = 2.0,
        dashboard_file: str = "wma_status_dashboard.log",
        history_file: str = "wma_dashboard_history.log",
        enabled: bool = True
    ):
        """
        Initialize the dashboard monitor.
        
        Args:
            participant_manager: Reference to ParticipantManager instance
            video_io_workers: List of video I/O workers
            video_api: Reference to VideoAPIManager instance
            update_interval: Seconds between updates (default: 2.0)
            dashboard_file: Path to live dashboard file (overwritten each update)
            history_file: Path to history file (append-only, timestamped snapshots)
            enabled: Whether monitoring is enabled
        """
        self.participant_manager = participant_manager
        self.video_io_workers = video_io_workers
        self.video_api = video_api
        self.update_interval = update_interval
        self.dashboard_file = Path(dashboard_file)
        self.history_file = Path(history_file)
        self.enabled = enabled
        
        self._stop_event = threading.Event()
        self._thread = None
        
        # Statistics tracking
        self.stats = {
            'updates_written': 0,
            'start_time': time.time(),
            'api_calls': 0,
            'api_timeouts': 0,
            'api_errors': 0
        }
        
        # Clear history file on initialization (fresh start)
        if self.enabled:
            try:
                self.history_file.write_text("", encoding='utf-8')
                logging.info(f"[Dashboard] History file cleared: {self.history_file}")
            except Exception as e:
                logging.error(f"[Dashboard] Failed to clear history file: {e}")
        
        logging.info(f"[Dashboard] Monitor initialized - interval={update_interval}s, enabled={enabled}")
    
    def start(self):
        """Start the monitoring thread."""
        if not self.enabled:
            logging.info("[Dashboard] Monitor disabled, not starting")
            return
        
        if self._thread is not None and self._thread.is_alive():
            logging.warning("[Dashboard] Monitor already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logging.info("[Dashboard] Monitor thread started")
    
    def stop(self):
        """Stop the monitoring thread."""
        if self._thread is None:
            return
        
        logging.info("[Dashboard] Stopping monitor thread...")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logging.info("[Dashboard] Monitor thread stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in background thread."""
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
        from rich.layout import Layout
        from io import StringIO
        
        while not self._stop_event.is_set():
            try:
                # Collect data
                participant_summaries = self.participant_manager.get_all_participant_summaries()
                manager_stats = self.participant_manager.get_manager_stats()
                
                # Generate dashboard content
                dashboard_content = self._generate_dashboard(
                    participant_summaries, 
                    manager_stats
                )
                
                # Write to live dashboard file (overwrite)
                try:
                    self.dashboard_file.write_text(dashboard_content, encoding='utf-8')
                except Exception as e:
                    logging.error(f"[Dashboard] Failed to write dashboard file: {e}")
                
                # Append timestamped snapshot to history file
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    history_entry = f"\n{'='*80}\n[SNAPSHOT] {timestamp}\n{'='*80}\n{dashboard_content}\n"
                    with open(self.history_file, 'a', encoding='utf-8') as f:
                        f.write(history_entry)
                except Exception as e:
                    logging.error(f"[Dashboard] Failed to append to history file: {e}")
                
                self.stats['updates_written'] += 1
                
            except Exception as e:
                logging.error(f"[Dashboard] Error in monitor loop: {e}")
            
            # Wait for next update
            self._stop_event.wait(self.update_interval)
    
    def _generate_dashboard(
        self, 
        participant_summaries: Dict[str, Dict[str, Any]],
        manager_stats: Dict[str, Any]
    ) -> str:
        """
        Generate formatted dashboard content using rich library.
        
        Args:
            participant_summaries: Dict of participant data
            manager_stats: Manager-level statistics
            
        Returns:
            Formatted dashboard string with ANSI color codes
        """
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
        from io import StringIO
        import wma_streaming_pb2 as pb2
        
        # Create a Console that writes to a string buffer
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True, width=120)
        
        # Header
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        uptime = time.time() - self.stats['start_time']
        uptime_str = self._format_duration(uptime)
        
        header = Text()
        header.append("WMA BACKEND LIVE STATUS DASHBOARD\n", style="bold white on blue")
        header.append(f"Last Updated: {now} | Server Uptime: {uptime_str}", style="dim")
        
        console.print(Panel(header, border_style="blue"))
        console.print()
        
        # Active Participants Table
        if participant_summaries:
            table = Table(
                title="🎥 ACTIVE PARTICIPANTS (Frame Processing)",
                show_header=True,
                header_style="bold cyan",
                border_style="cyan",
                title_style="bold cyan"
            )
            
            table.add_column("Participant ID", style="white", no_wrap=True, width=20)
            table.add_column("Status", justify="center", width=8)
            table.add_column("Queue", justify="right", width=6)
            table.add_column("Last Frame", justify="right", width=11)
            table.add_column("Mean Prob", justify="right", width=10)
            table.add_column("Verdict", justify="center", width=8)
            table.add_column("Trend (Last 10)", width=25)
            table.add_column("Batches", justify="right", width=8)
            
            # Sort by last seen (most recent first)
            sorted_participants = sorted(
                participant_summaries.items(),
                key=lambda x: x[1]['last_seen_ts'],
                reverse=True
            )
            
            for pid, data in sorted_participants:
                # Status with color
                if data['is_active']:
                    status = Text("ACTIVE", style="bold green")
                else:
                    status = Text("IDLE", style="dim yellow")
                
                # Time since last frame
                last_frame = self._format_time_ago(data['inactive_seconds'])
                
                # Mean probability with color coding
                mean_prob = data['mean_prob']
                if mean_prob >= self.video_api.threshold + self.participant_manager.margin:
                    prob_text = Text(f"{mean_prob:.3f}", style="bold red")
                elif mean_prob >= self.video_api.threshold - self.participant_manager.margin:
                    prob_text = Text(f"{mean_prob:.3f}", style="bold yellow")
                else:
                    prob_text = Text(f"{mean_prob:.3f}", style="bold green")
                
                # Verdict with color
                verdict_level = data['current_verdict']
                verdict_name = pb2.BannerLevel.Name(verdict_level)
                if verdict_level == pb2.RED:
                    verdict_text = Text(verdict_name, style="bold white on red")
                elif verdict_level == pb2.YELLOW:
                    verdict_text = Text(verdict_name, style="bold black on yellow")
                else:
                    verdict_text = Text(verdict_name, style="bold white on green")
                
                # ASCII Sparkline
                sparkline = self._generate_sparkline(data['history'])
                
                # Queue size
                queue_size = data['full_history_size']
                
                # Batch counter
                batches = str(data['batch_counter'])
                
                table.add_row(
                    pid[:20],  # Truncate long IDs
                    status,
                    str(queue_size),
                    last_frame,
                    prob_text,
                    verdict_text,
                    sparkline,
                    batches
                )
            
            console.print(table)
            console.print()
            
            # Detailed Participant Info (top 3 active)
            active_participants = [
                (pid, data) for pid, data in sorted_participants 
                if data['is_active']
            ][:3]  # Top 3
            
            if active_participants:
                console.print("[bold white]📊 PARTICIPANT DETAILS (Top 3 Active)[/bold white]")
                console.print()
                
                for pid, data in active_participants:
                    self._print_participant_detail(console, pid, data)
        else:
            console.print("[dim]No active participants[/dim]")
            console.print()
        
        # System Stats
        self._print_system_stats(console, manager_stats)
        
        # Get the formatted output
        return buffer.getvalue()
    
    def _print_participant_detail(self, console, pid: str, data: Dict[str, Any]):
        """Print detailed information for a single participant."""
        from rich.text import Text
        import wma_streaming_pb2 as pb2
        
        verdict_name = pb2.BannerLevel.Name(data['current_verdict'])
        
        # Confidence calculation (simplified, matches server logic)
        mean_prob = data['mean_prob']
        thr = self.video_api.threshold
        m = self.participant_manager.margin
        
        if data['current_verdict'] == pb2.YELLOW:
            confidence_label = "Uncertain"
        elif data['current_verdict'] == pb2.GREEN:
            # Lower score = higher confidence it's real
            green_max = thr - m
            if mean_prob < green_max * 0.5:
                confidence_label = "High"
            elif mean_prob < green_max * 0.75:
                confidence_label = "Medium"
            else:
                confidence_label = "Low"
        else:  # RED
            # Higher score = higher confidence it's fake
            red_min = thr + m
            red_range = 1.0 - red_min
            relative = mean_prob - red_min
            if relative < red_range * 0.33:
                confidence_label = "Low"
            elif relative < red_range * 0.66:
                confidence_label = "Medium"
            else:
                confidence_label = "High"
        
        detail = Text()
        detail.append(f"  {pid}\n", style="bold white")
        detail.append(f"    History Window: {data['history'][:5]} (first 5 of {data['full_history_size']})\n", style="dim")
        detail.append(f"    Batch Counter: {data['batch_counter']}\n", style="dim")
        detail.append(f"    Frames Processed: {data['frames_processed']}\n", style="dim")
        detail.append(f"    Current Confidence: {confidence_label}\n", style="cyan")
        detail.append(f"    Verdict: {verdict_name}\n", style="yellow" if data['current_verdict'] == pb2.YELLOW else ("red" if data['current_verdict'] == pb2.RED else "green"))
        
        console.print(detail)
        console.print()
    
    def _print_system_stats(self, console, manager_stats: Dict[str, Any]):
        """Print system-level statistics."""
        from rich.table import Table
        
        console.print("[bold white]⚙️  SYSTEM STATISTICS[/bold white]")
        console.print()
        
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Active Participants", str(manager_stats['active_participants']))
        stats_table.add_row("Total Participants", str(manager_stats['total_participants']))
        stats_table.add_row("Detection Threshold", f"{manager_stats['threshold']:.3f}")
        stats_table.add_row("Margin", f"{manager_stats['margin']:.3f}")
        stats_table.add_row("History Window Size", str(manager_stats['history_window_size']))
        stats_table.add_row("Active Test Window", str(manager_stats['active_test_window']))
        
        # I/O Worker queue sizes
        if self.video_io_workers:
            for i, worker in enumerate(self.video_io_workers, 1):
                queue_size = worker.q.qsize()
                stats_table.add_row(f"Video Worker {i} Queue", str(queue_size))
        
        stats_table.add_row("Dashboard Updates", str(self.stats['updates_written']))
        
        console.print(stats_table)
    
    def _generate_sparkline(self, history: List[float]) -> str:
        """
        Generate ASCII sparkline from probability history.
        
        Args:
            history: List of probabilities (most recent first)
            
        Returns:
            ASCII sparkline string with trend indicator
        """
        if not history or len(history) < 2:
            return "─" * 10
        
        # Use up to 10 most recent values
        values = history[:10]
        
        # Sparkline characters (from lowest to highest)
        chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        # Normalize to 0-1 range
        min_val = 0.0
        max_val = 1.0
        
        sparkline = ""
        for val in reversed(values):  # Reverse to show oldest -> newest
            # Map value to character index
            normalized = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            char_idx = int(normalized * (len(chars) - 1))
            char_idx = max(0, min(len(chars) - 1, char_idx))
            sparkline += chars[char_idx]
        
        # Add trend indicator
        if len(values) >= 3:
            recent_avg = sum(values[:3]) / 3
            older_avg = sum(values[3:6]) / 3 if len(values) >= 6 else recent_avg
            
            if recent_avg > older_avg + 0.05:
                trend = " ↗"
            elif recent_avg < older_avg - 0.05:
                trend = " ↘"
            else:
                trend = " →"
        else:
            trend = ""
        
        return sparkline + trend
    
    def _format_time_ago(self, seconds: float) -> str:
        """Format seconds into human-readable 'time ago' string."""
        if seconds < 1:
            return "now"
        elif seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m ago"
        else:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def update_api_stats(self, success: bool, timeout: bool = False):
        """Update API call statistics (called from video API manager)."""
        self.stats['api_calls'] += 1
        if timeout:
            self.stats['api_timeouts'] += 1
        if not success:
            self.stats['api_errors'] += 1
```

### Why This Design:

1. **Separate Thread** - No impact on main processing loop
2. **Rich Library** - Beautiful colors and formatting that works over SSH
3. **Sparklines** - Visual representation of trends at a glance
4. **Two Files** - Live view + historical archive
5. **Minimal Locking** - Quick data collection, formatting happens outside locks
6. **Error Handling** - Continues running even if file writes fail

---

## **TASK 4: Integrate Dashboard into StreamingServiceImpl**

**File:** `server.py`

**Location:** In the `StreamingServiceImpl.__init__` method (around line 550)

### What to modify:

**BEFORE** (around line 600, after participant_manager initialization):
```python
        # --- Participant name matcher ---
        # Initialize with configurable sensitivity (can be adjusted via env var)
        name_match_threshold = float(os.getenv("WMA_NAME_MATCH_THRESHOLD", "0.3"))
        self.name_matcher = ParticipantNameMatcher(similarity_threshold=name_match_threshold)
        logging.info(f"[Backend] Participant name matching threshold: {name_match_threshold}")
```

**AFTER** (add this right after the name_matcher initialization):
```python
        # --- Participant name matcher ---
        # Initialize with configurable sensitivity (can be adjusted via env var)
        name_match_threshold = float(os.getenv("WMA_NAME_MATCH_THRESHOLD", "0.3"))
        self.name_matcher = ParticipantNameMatcher(similarity_threshold=name_match_threshold)
        logging.info(f"[Backend] Participant name matching threshold: {name_match_threshold}")
        
        # --- Status Dashboard Monitor ---
        dashboard_enabled = os.getenv("WMA_DASHBOARD_ENABLED", "false").lower() == "true"
        dashboard_interval = float(os.getenv("WMA_DASHBOARD_INTERVAL", "2.0"))
        dashboard_file = os.getenv("WMA_DASHBOARD_FILE", "wma_status_dashboard.log")
        history_file = os.getenv("WMA_DASHBOARD_HISTORY", "wma_dashboard_history.log")
        
        self.dashboard_monitor = StatusDashboardMonitor(
            participant_manager=self.participant_manager,
            video_io_workers=self.video_io_workers,
            video_api=self.video_api,
            update_interval=dashboard_interval,
            dashboard_file=dashboard_file,
            history_file=history_file,
            enabled=dashboard_enabled
        )
        
        if dashboard_enabled:
            logging.info(f"[Backend] Dashboard monitor enabled - interval={dashboard_interval}s")
            logging.info(f"[Backend] Dashboard file: {dashboard_file}")
            logging.info(f"[Backend] History file: {history_file}")
        else:
            logging.info("[Backend] Dashboard monitor disabled (set WMA_DASHBOARD_ENABLED=true to enable)")
```

### Also modify the `cleanup` method:

**BEFORE** (around line 1400):
```python
    def cleanup(self):
        """Clean up resources."""
        logging.info("[Backend] Cleaning up I/O workers...")
        for worker in self.video_io_workers + self.audio_io_workers:
            worker.stop()
        
        # Reset audio window manager
        self.audio_window_manager.reset()
        logging.info("[Backend] Audio window manager reset")
```

**AFTER**:
```python
    def cleanup(self):
        """Clean up resources."""
        logging.info("[Backend] Cleaning up I/O workers...")
        for worker in self.video_io_workers + self.audio_io_workers:
            worker.stop()
        
        # Reset audio window manager
        self.audio_window_manager.reset()
        logging.info("[Backend] Audio window manager reset")
        
        # Stop dashboard monitor
        if hasattr(self, 'dashboard_monitor'):
            self.dashboard_monitor.stop()
```

### And modify the `serve()` function to start the dashboard:

**BEFORE** (around line 1600, after `service_impl = StreamingServiceImpl()`):
```python
    # Add service
    service_impl = StreamingServiceImpl()
    pb2_grpc.add_StreamingServiceServicer_to_server(service_impl, server)
```

**AFTER**:
```python
    # Add service
    service_impl = StreamingServiceImpl()
    pb2_grpc.add_StreamingServiceServicer_to_server(service_impl, server)
    
    # Start dashboard monitor if enabled
    if hasattr(service_impl, 'dashboard_monitor'):
        service_impl.dashboard_monitor.start()
```

### Why:
- **Environment variables** - Easy to enable/disable without code changes
- **Configurable** - Interval, file paths can be customized
- **Clean lifecycle** - Started after init, stopped on shutdown
- **Backwards compatible** - Disabled by default, doesn't break existing deployments

---

## **TASK 5: Add Command-Line Arguments**

**File:** `start_backend.py`

**Location:** In the `DEFAULTS` dict (around line 30)

### What to modify:

**BEFORE**:
```python
DEFAULTS = {
    # Video API configuration
    "VIDEO_API_HOST": "34.16.217.28",
    "VIDEO_API_PORT": "8999",
    "VIDEO_API_TIMEOUT": "30",
    "VIDEO_YOLO_CONF_THRESHOLD": "0.80",

    # Inference defaults for WMA backend
    "WMA_INFER_THRESHOLD": "0.85",
    "WMA_INFER_BATCH": "16",
    "WMA_BAND_MARGIN": "0.05",
}
```

**AFTER**:
```python
DEFAULTS = {
    # Video API configuration
    "VIDEO_API_HOST": "34.16.217.28",
    "VIDEO_API_PORT": "8999",
    "VIDEO_API_TIMEOUT": "30",
    "VIDEO_YOLO_CONF_THRESHOLD": "0.80",

    # Inference defaults for WMA backend
    "WMA_INFER_THRESHOLD": "0.85",
    "WMA_INFER_BATCH": "16",
    "WMA_BAND_MARGIN": "0.05",
    
    # Dashboard configuration (NEW)
    "WMA_DASHBOARD_ENABLED": "false",
    "WMA_DASHBOARD_INTERVAL": "2.0",
    "WMA_DASHBOARD_FILE": "wma_status_dashboard.log",
    "WMA_DASHBOARD_HISTORY": "wma_dashboard_history.log",
}
```

### Also modify the `parse_args()` function in `server.py`:

**BEFORE** (around line 1440):
```python
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='WMA Backend gRPC Server')

    parser.add_argument('--video-bucket', type=str, default=None,
                        help='GCP bucket name for video frames')
    parser.add_argument('--audio-bucket', type=str, default=None,
                        help='GCP bucket name for audio chunks')
    parser.add_argument('--enable-bucket-save', action='store_true',
                        help='Enable saving to GCP buckets (default: disabled)')
    parser.add_argument('--io-workers', type=int, default=2,
                        help='Number of I/O workers per media type (default: 2)')
    parser.add_argument('--port', type=int, default=50051,
                        help='gRPC server port (default: 50051)')

    parser.add_argument('--debug', action='store_true',
                        help='Enable verbose debug logging')

    return parser.parse_args()
```

**AFTER**:
```python
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='WMA Backend gRPC Server')

    parser.add_argument('--video-bucket', type=str, default=None,
                        help='GCP bucket name for video frames')
    parser.add_argument('--audio-bucket', type=str, default=None,
                        help='GCP bucket name for audio chunks')
    parser.add_argument('--enable-bucket-save', action='store_true',
                        help='Enable saving to GCP buckets (default: disabled)')
    parser.add_argument('--io-workers', type=int, default=2,
                        help='Number of I/O workers per media type (default: 2)')
    parser.add_argument('--port', type=int, default=50051,
                        help='gRPC server port (default: 50051)')

    parser.add_argument('--debug', action='store_true',
                        help='Enable verbose debug logging')
    
    # Dashboard arguments (NEW)
    parser.add_argument('--dashboard-enabled', action='store_true',
                        help='Enable live status dashboard (default: disabled)')
    parser.add_argument('--dashboard-interval', type=float, default=2.0,
                        help='Dashboard update interval in seconds (default: 2.0)')
    parser.add_argument('--dashboard-file', type=str, default='wma_status_dashboard.log',
                        help='Dashboard output file (default: wma_status_dashboard.log)')
    parser.add_argument('--dashboard-history', type=str, default='wma_dashboard_history.log',
                        help='Dashboard history file (default: wma_dashboard_history.log)')

    return parser.parse_args()
```

### And update the environment variable setting in `serve()`:

**BEFORE** (around line 1470, in `serve()` function):
```python
    # Set global variables
    global GCP_VIDEO_BUCKET, GCP_AUDIO_BUCKET, ENABLE_BUCKET_SAVE, IO_WORKER_COUNT, DEBUG_MODE
    GCP_VIDEO_BUCKET = args.video_bucket
    GCP_AUDIO_BUCKET = args.audio_bucket
    ENABLE_BUCKET_SAVE = args.enable_bucket_save
    IO_WORKER_COUNT = args.io_workers
    DEBUG_MODE = args.debug
```

**AFTER**:
```python
    # Set global variables
    global GCP_VIDEO_BUCKET, GCP_AUDIO_BUCKET, ENABLE_BUCKET_SAVE, IO_WORKER_COUNT, DEBUG_MODE
    GCP_VIDEO_BUCKET = args.video_bucket
    GCP_AUDIO_BUCKET = args.audio_bucket
    ENABLE_BUCKET_SAVE = args.enable_bucket_save
    IO_WORKER_COUNT = args.io_workers
    DEBUG_MODE = args.debug
    
    # Set dashboard environment variables
    os.environ['WMA_DASHBOARD_ENABLED'] = 'true' if args.dashboard_enabled else 'false'
    os.environ['WMA_DASHBOARD_INTERVAL'] = str(args.dashboard_interval)
    os.environ['WMA_DASHBOARD_FILE'] = args.dashboard_file
    os.environ['WMA_DASHBOARD_HISTORY'] = args.dashboard_history
```

### Why:
- **Flexibility** - Can enable via environment variable OR command line
- **Defaults** - Sensible defaults that match requirements
- **Backwards compatible** - Disabled by default

---

## **TASK 6: Update Requirements**

**File:** `requirements.txt`

Add:
```
rich>=13.0.0
```

Then run:
```bash
pip install -r requirements.txt
```

---

## 🧪 Testing Plan

### Phase 1: Basic Functionality
```bash
# Start server with dashboard enabled
python start_backend.py --dashboard-enabled

# In another terminal, watch the live dashboard
tail -f wma_status_dashboard.log

# Verify:
# - Dashboard updates every 2 seconds
# - Participant table shows correct data
# - Colors display properly
# - Sparklines render correctly
```

### Phase 2: Performance Testing
```bash
# Monitor CPU usage with dashboard running
top -p $(pgrep -f start_backend)

# Verify:
# - CPU usage increase is minimal (<5%)
# - No impact on processing latency
# - File I/O doesn't block main thread
```

### Phase 3: Edge Cases
```bash
# Test with no participants
# - Dashboard should show "No active participants"

# Test with many participants (10+)
# - Verify table doesn't overflow
# - Performance remains stable

# Test with long participant names
# - Verify truncation works correctly
```

### Phase 4: History File
```bash
# Let server run for 5+ minutes
# Check history file has timestamped snapshots
cat wma_dashboard_history.log | grep SNAPSHOT

# Restart server
# Verify history file is cleared on restart
```

---

## 📊 Expected Output Examples

### Live Dashboard (`wma_status_dashboard.log`):

```
╭─────────────────────────────────────────────────────────────────╮
│ WMA BACKEND LIVE STATUS DASHBOARD                              │
│ Last Updated: 2025-11-02 14:23:45.123 | Server Uptime: 1h 23m │
╰─────────────────────────────────────────────────────────────────╯

           🎥 ACTIVE PARTICIPANTS (Frame Processing)            
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Participant ID       ┃ Status ┃ Queue┃ Last Frame┃ Mean Prob┃ Verdict┃ Trend (Last 10)         ┃ Batches┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Alice Martinez       │ ACTIVE │   48 │    1s ago │   0.230  │ GREEN  │ ▂▂▂▁▂▂▃▂▂▁ →           │      3 │
│ Bob Johnson          │ ACTIVE │   48 │    2s ago │   0.875  │  RED   │ ▆▇█▇█▇▆▇█▇ ↗           │      1 │
│ Charlie Wong         │ ACTIVE │   48 │    3s ago │   0.790  │ YELLOW │ ▅▅▅▆▅▅▅▆▅▅ →           │      5 │
│ David_unknown        │  IDLE  │   48 │   45s ago │   0.150  │ GREEN  │ ▁▁▂▁▁▁▁▂▁▁ ↘           │      0 │
└──────────────────────┴────────┴──────┴───────────┴──────────┴────────┴─────────────────────────┴────────┘

📊 PARTICIPANT DETAILS (Top 3 Active)

  Alice Martinez
    History Window: [0.22, 0.23, 0.21, 0.24, 0.22] (first 5 of 48)
    Batch Counter: 3
    Frames Processed: 48
    Current Confidence: High
    Verdict: GREEN

  Bob Johnson
    History Window: [0.88, 0.87, 0.89, 0.86, 0.87] (first 5 of 48)
    Batch Counter: 1
    Frames Processed: 45
    Current Confidence: High
    Verdict: RED

⚙️  SYSTEM STATISTICS

  Active Participants      2
  Total Participants       4
  Detection Threshold      0.850
  Margin                   0.050
  History Window Size      48
  Active Test Window       20
  Video Worker 1 Queue     0
  Video Worker 2 Queue     0
  Dashboard Updates        245
```

### History File (`wma_dashboard_history.log`):

```
================================================================================
[SNAPSHOT] 2025-11-02 14:23:43
================================================================================
[Full dashboard content here...]

================================================================================
[SNAPSHOT] 2025-11-02 14:23:45
================================================================================
[Full dashboard content here...]
```

---

## ⚠️ Potential Issues & Solutions

### Issue 1: Rich Colors Don't Show Over SSH
**Symptom:** ANSI codes display as raw text  
**Solution:** Ensure terminal supports ANSI colors:
```bash
export TERM=xterm-256color
tail -f wma_status_dashboard.log
```

### Issue 2: Dashboard Updates Too Fast/Slow
**Symptom:** Hard to read or too stale  
**Solution:** Adjust interval via CLI:
```bash
python start_backend.py --dashboard-enabled --dashboard-interval 5.0
```

### Issue 3: File Grows Too Large
**Symptom:** History file becomes huge over time  
**Solution:** History file is cleared on server restart (by design). For long-running servers, add rotation:
```python
# In StatusDashboardMonitor.__init__:
if self.history_file.stat().st_size > 100_000_000:  # 100MB
    self.history_file.write_text("", encoding='utf-8')
```

### Issue 4: Performance Impact
**Symptom:** Processing slowdown with dashboard enabled  
**Solution:**
- Increase update interval (5s or 10s)
- Reduce amount of detail in output
- Check for lock contention (add timing logs)

### Issue 5: Participant Names Truncated
**Symptom:** Long names cut off  
**Solution:** Adjust column width in table definition:
```python
table.add_column("Participant ID", style="white", no_wrap=True, width=30)  # Increase width
```

---

## 🚀 Deployment Checklist

- [ ] Install `rich` library: `pip install rich>=13.0.0`
- [ ] Add methods to `ParticipantManager`: `get_all_participant_summaries()`, `get_manager_stats()`
- [ ] Add `StatusDashboardMonitor` class to `server.py`
- [ ] Integrate monitor into `StreamingServiceImpl.__init__()`
- [ ] Start monitor in `serve()` function
- [ ] Stop monitor in `cleanup()` method
- [ ] Add CLI arguments to `parse_args()`
- [ ] Update `DEFAULTS` in `start_backend.py`
- [ ] Test with dashboard disabled (default behavior)
- [ ] Test with dashboard enabled
- [ ] Verify colors work over SSH
- [ ] Verify sparklines render correctly
- [ ] Verify history file is created and appended
- [ ] Verify history file clears on restart
- [ ] Check performance impact (should be <5% CPU)

---

## 📞 Usage Examples

### Enable Dashboard (Recommended):
```bash
python start_backend.py --dashboard-enabled
```

### Enable with Custom Interval:
```bash
python start_backend.py --dashboard-enabled --dashboard-interval 5.0
```

### Enable with Custom File Paths:
```bash
python start_backend.py --dashboard-enabled \
    --dashboard-file /var/log/wma/dashboard.log \
    --dashboard-history /var/log/wma/history.log
```

### View Live Dashboard:
```bash
# Terminal 1: Run server
python start_backend.py --dashboard-enabled

# Terminal 2: Watch dashboard
tail -f wma_status_dashboard.log

# Terminal 3: Search history
grep "Bob Johnson" wma_dashboard_history.log
```

### Enable via Environment Variable:
```bash
export WMA_DASHBOARD_ENABLED=true
export WMA_DASHBOARD_INTERVAL=2.0
python start_backend.py
```

---

## 🎓 Code Quality Guidelines

1. **Thread Safety:**
   - All access to `ParticipantManager.participants` must use `self.lock`
   - Dashboard thread minimizes lock time by collecting data quickly
   - Formatting happens outside locks

2. **Error Handling:**
   - All file I/O wrapped in try-except
   - Dashboard failures don't crash server
   - Errors logged but processing continues

3. **Performance:**
   - Dashboard updates are non-blocking
   - File writes are buffered
   - No synchronous I/O in main processing thread

4. **Logging:**
   - Use `logging.info()` for status messages
   - Use `logging.error()` for failures
   - Don't spam logs with dashboard updates

5. **Testing:**
   - Test with 0, 1, 5, 10+ participants
   - Test with very long participant names
   - Test with rapid verdict changes
   - Test server restart behavior

---

## 📚 Additional Resources

- **Rich Library Docs:** https://rich.readthedocs.io/
- **Sparklines Reference:** https://en.wikipedia.org/wiki/Sparkline
- **ANSI Color Codes:** https://en.wikipedia.org/wiki/ANSI_escape_code

---

## ✅ Success Criteria

When implementation is complete, you should be able to:

1. ✅ Start server with `--dashboard-enabled` flag
2. ✅ Open another terminal and run `tail -f wma_status_dashboard.log`
3. ✅ See a **color-coded table** updating every 2 seconds
4. ✅ See participant status, probabilities, and verdicts
5. ✅ See **ASCII sparklines** showing probability trends
6. ✅ See trend indicators (↗↘→) showing if probabilities are increasing/decreasing
7. ✅ See detailed info for top 3 active participants
8. ✅ See system stats at the bottom
9. ✅ Verify `wma_dashboard_history.log` contains timestamped snapshots
10. ✅ Restart server and verify history file is cleared

---

**This implementation provides excellent real-time visibility with minimal overhead and follows best practices for monitoring production systems.**
