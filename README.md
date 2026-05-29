# Real-Time Webcam Pen-Tip Tracker

Hybrid pipeline: **PenTipTrack** (siamese NN) + **Lucas–Kanade optical flow** + **Kalman 3D** + monocular relative depth. Tracks the pen tip from a live webcam, draws a depth-colored trajectory, records MP4, and writes per-frame JSONL logs.

**Assignment spec:** [Test Assignment Real-Time Webcam.md](Test%20Assignment%20Real-Time%20Webcam.md)

**Documentation:**

- [Assignment report](docs/REPORT.md) — architecture, failure cases, future work

---

## Requirements

- Python 3.10+
- Webcam
- Windows / Linux / macOS (DirectShow backend on Windows)

**Python packages** (`requirements.txt`):

```text
numpy==2.2.6
opencv-python==4.13.0.92
torch==2.12.0
torchvision==0.27.0
yacs==0.1.8
```

(Full list: [requirements.txt](requirements.txt). Pinned on Python 3.10, Windows; use matching CUDA wheels for GPU if needed.)

CUDA is used automatically when available; CPU fallback works but NN is slower.

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url> tracking_proj
cd tracking_proj
pip install -r requirements.txt
```

### 2. Model weights

Pretrained PenTipTrack weights are included in the repository:

| File | Version |
|------|---------|
| `pentiptrackv2.pth` | V2 |
| `pentiptrackv3.pth` | V3 (default CLI) |

If either file is missing, the app exits with `Missing weights: models/pretrained/...`.

NN configs live in `tracker/nn/configs/` (`configv2.yaml`, `configv3.yaml`).

---

## Run

```bash
python run_webcam_tracker.py
```

### Common options

```bash
# V3 model, NN every 3 frames, camera 0, save video + log
python run_webcam_tracker.py \
  --pentiptrack-version v3 \
  --nn-interval 3 \
  --camera 0 \
  --output demo_output.mp4 \
  --log tracking_log.jsonl

# Debug: HUD on screen + extended JSONL fields
python run_webcam_tracker.py --debug

# Limit session length (0 = until quit)
python run_webcam_tracker.py --max-frames 900

# Request capture resolution (0 = camera maximum, default)
python run_webcam_tracker.py --width 1280 --height 720
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Webcam device index |
| `--output` | `demo_output.mp4` | Recorded overlay video |
| `--log` | `tracking_log.jsonl` | Per-frame JSONL log |
| `--max-frames` | `0` | Stop after N frames (0 = unlimited) |
| `--width` / `--height` | `0` | Capture size; `0` = max native |
| `--pentiptrack-version` | `v3` | `v2` or `v3` |
| `--nn-interval` | `3` | Run NN every N frames |
| `--debug` | off | Debug HUD + extra log fields |

---

## Initialization & controls

1. **Start** — window opens on the first camera frame.
2. **Select target:**
   - **Drag** LMB — rectangular ROI around the pen tip.
   - **Click** LMB — quick square box (5% of the shorter frame side, centered on click).
3. Tracking starts immediately after ROI selection.

**During tracking:**

| Input | Action |
|-------|--------|
| `r` | Manual re-select ROI |
| `q` / `ESC` | Quit |
| *(automatic)* | On `lost`, ROI selection opens again |

---

## Outputs

### Video (`--output`)

Each frame shows:

- Pen-tip crosshair (Kalman position)
- Bounding box (color = `visible` / `uncertain` / `lost`)
- Trajectory polyline — **color and thickness encode relative depth** (closer → thicker, warmer/red)

With `--debug`, also: NN search ROI, LK flow ROI, flow vectors, LK feature inliers (green) / outliers (red), latency HUD.

### Log (`--log`)

One JSON object per line. **Required fields** (always):

```json
{
  "frame": 127,
  "timestamp_ms": 4233.2,
  "x": 318.4,
  "y": 241.9,
  "z_relative": 0.63,
  "confidence": 0.82,
  "tracking_state": "visible",
  "method_used": "pentiptrack_v2",
  "latency_ms": 18.7
}
```

`tracking_state`: `visible` | `uncertain` | `lost` | `reacquired`

`method_used`: `pentiptrack_v2` / `pentiptrack_v3` | `optical_flow` | `kalman_prediction`

With `--debug`, additional timing and diagnostic fields (`nn_ms`, `cv_ms`, `nn_peak_score`, …).

### Console summary (end of session)

Printed after quit:

- Session FPS (wall clock)
- Processing FPS (pipeline only)
- Avg / P95 / P99 latency (ms)
- Estimated dropped camera frames
- NN inference rate

---

## Project layout

```text
run_webcam_tracker.py      # entry point
app/                       # webcam session, UI, telemetry, recording
tracker/
  tracking/                # pipeline, fusion, FSM, frame tracker
  backends/pentiptrack.py  # NN localizer
  cv/local_flow.py         # LK optical flow
  motion/kalman.py         # Kalman [x,y,z,vx,vy,vz]
  depth/                   # relative depth from NN bbox scale
  nn/                      # PenTipTrack model code
models/pretrained/         # PenTipTrack v2/v3 weights
docs/                      # assignment report (REPORT.md)
```

Canonical API:

```python
from tracker.tracking import TrackingPipeline, create_tracking_pipeline
```

---

## Tuning

Key thresholds in `tracker/config.py` (`TrackingConfig`):

- `nn_interval` — NN rate and lost-streak tolerance (via FSM)
- `peak_reinit_threshold` / `peak_lost_threshold` — NN peak gating
- `flow_outlier_mad_k` — LK displacement outlier rejection
- Kalman noise (`kalman_r_*`, `kalman_q_*`) — smoothness vs responsiveness

Override at factory time:

```python
create_tracking_pipeline(fps=30, nn_interval=2, pentiptrack_version="v3")
```

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `Missing weights` | Ensure `models/pretrained/*.pth` are present (shipped in repo) |
| Low Session FPS | Normal if MP4 + imshow dominate; see Processing FPS |
| Frequent `lost` | Lower `--nn-interval`, better lighting, larger init ROI |
| Jittery bbox | Expected between NN frames; Kalman smooths tip position |
| `confidence` stuck high on flow frames | Last NN confidence is held; use `method_used` in log |

See [docs/REPORT.md](docs/REPORT.md) for architecture and detailed failure modes.
