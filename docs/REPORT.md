# Architecture Report — Real-Time Pen-Tip Tracking

This document describes the **current** production pipeline in `tracking_proj`: design rationale, component roles, data flow, and key tradeoffs. It matches the hybrid NN + classical CV + Kalman architecture required by the assignment.

---

## 1. Design goals

1. **Low latency** — NN does not run every frame; classical CV bridges gaps.
2. **Robustness** — periodic NN correction when LK drifts or confidence drops.
3. **Smooth output** — Kalman filter on `(x, y, z)` with velocity state.
4. **Monocular depth cue** — relative `z` from apparent object scale (NN bbox vs init ROI).
5. **Measurable** — per-frame JSONL, pipeline timing breakdown, session metrics.

---

## 2. High-level pipeline

```text
Webcam frame
    ↓
Timestamp (wall clock ms)
    ↓
Kalman predict  →  (x_pred, y_pred, z_pred)
    ↓
LK optical flow (ROI = NN bbox size)  →  flow (dx, dy), inlier features
    ↓
HybridFrameTracker.track()
    ├─ NnScheduler: run PenTipTrack? (interval / low peak)
    └─ PenTipTrack inference (when scheduled)
    ↓
PeakTrust + Fusion
    ├─ Trusted NN  → update Kalman + depth from bbox
    ├─ Valid flow  → Kalman bridge (if peak not dead)
    └─ Else        → predict only, decay velocity
    ↓
TrackingStateMachine  →  visible | uncertain | lost | reacquired
    ↓
Display bbox from Kalman center + NN width/height
    ↓
Overlay + trajectory + MP4 + JSONL
```

Entry point: `app/session.py` → `tracker/tracking/pipeline.py`.

---

## 3. Why this architecture

| Alternative | Issue | My choice |
|-------------|-------|------------|
| NN every frame | Too slow on CPU / embedded | NN every `N` frames + LK between |
| LK only | Drifts on clutter, blur | PenTipTrack re-anchors |
| Template matching on pixels | Jumps to similar background | Removed; siamese NN used instead |
| LK for depth scale | Unstable on approach; latched to background | **Depth only from NN bbox** (current) |
| Offline post-processing | Violates real-time requirement | All outputs from live loop |

The project evolved through several prototypes (template matching, Siamese NCC, LK-only scale). The stable stack is **PenTipTrack + LK motion + Kalman + bbox-based depth**.

---

## 4. Component details

### 4.1 Neural network — PenTipTrack V2/V3

**Role:** Robust 2D localization and bbox size; periodic correction.

- Implementation: `tracker/backends/pentiptrack.py`, `tracker/nn/*`
- Siamese-style tracker with MobileNetV3 backbone, correlation head
- **Initialization:** user ROI → template / exemplar on first frame
- **Outputs:** tip `(x, y)`, confidence, peak score, `width`, `height`
- **Schedule:** `NnScheduler` — frame 1, every `nn_interval` frames, or when `last_peak_score < nn_force_peak`
- **Online update:** EMA exemplar refresh when confidence high (`maybe_update_neural_template`)

Peak score gates trust (`PeakTrustTracker`): measurement must pass `peak_reinit_threshold`, confidence threshold, and not drop too far below peak EMA.

### 4.2 Classical CV — Lucas–Kanade optical flow

**Role:** Fast frame-to-frame motion between NN runs.

- Implementation: `tracker/cv/local_flow.py`
- `goodFeaturesToTrack` inside ROI matching **current NN bbox size**
- Pyramidal LK (`calcOpticalFlowPyrLK`, 3 levels)
- **Robust median** displacement over inlier features only
- **Outlier rejection:** MAD on per-feature displacement vs median (`flow_outlier_mad_k`, `flow_outlier_min_px`)
- Flow shifts NN search center: `search = motion_xy + flow * flow_pred_blend`

Flow updates Kalman only when: `flow_valid`, peak ≥ `peak_lost_threshold`, and fusion confidence thresholds met. It does **not** reset the lost-state streak (only trusted NN does).

### 4.3 Kalman filter — constant velocity 3D

**Role:** Prediction, smoothing, short-gap bridging.

- Implementation: `tracker/motion/kalman.py`
- **State vector:** `[x, y, z, vx, vy, vz]`
- **Measurements:** `(x, y, z)` from NN or flow-corrected position
- Measurement noise scaled by `1/confidence`
- `sync_velocity()` nudges velocity after NN/flow updates
- `decay_velocity()` when no valid measurement

`z` in Kalman tracks smoothed relative depth; raw scale comes from `RelativeDepthEstimator`.

### 4.4 Relative depth

**Role:** Monocular depth proxy for trajectory visualization and logging.

- Implementation: `tracker/depth/relative.py`, `scale_limits.py`
- At init: store ROI `(w, h)` as reference
- On trusted NN frames:  
  `scale = sqrt((w/w₀)·(h/h₀))` with rate limits (±30% retreat, +45% approach per step)
- `z_relative = clip(0.5 + 0.5·(scale − 1), 0, 1)` for JSONL
- **Trajectory color/thickness** uses `object_scale_ratio` with 2× visual gain

No metric calibration — perceived depth only.

### 4.5 Measurement fusion

**Priority** (`tracker/tracking/fusion.py`):

1. **Trusted NN** — update Kalman + `depth.observe_bbox(w, h)`; method = `pentiptrack_v*`
2. **Else valid flow** (peak not dead) — Kalman at `pred + flow`; method = `optical_flow`
3. **Else** — Kalman predict only; decay velocity; method = `kalman_prediction`

Display bbox: center from **Kalman** `(x, y)`, size from **last NN** `(width, height)` — decoupled so LK does not resize the box.

### 4.6 Tracking state machine

**Implementation:** `tracker/tracking/policy/state_machine.py`

| State | Meaning |
|-------|---------|
| `visible` | Trusted measurement, conf ≥ tracking threshold |
| `uncertain` | Weak or missing measurement, streak building |
| `lost` | Streak exceeded, bad peaks, or peak below reinit threshold |
| `reacquired` | One frame after manual ROI re-select |

**Lost triggers:**

- `_lost_streak > nn_interval` without valid NN-trusted measurement
- `_bad_peak_streak ≥ 2` on NN frames without valid measurement
- After warmup: `nn_peak_score < peak_reinit_threshold` on an NN frame

On `lost`, `app/session.py` pauses tracking and opens ROI selector again.

---

## 5. Initialization

**Method:** interactive ROI (`app/ui/roi.py`)

- Drag rectangle **or** single click → 5% side quick box
- Limitations: user must point at tip; no auto-detect; quick box assumes tip near click center
- Re-init: automatic on lost, manual via `r`

Pipeline `initialize()` resets: frame tracker, depth scale, Kalman, FSM, peak EMA.

---

## 6. Visualization & logging

| Layer | Module | Notes |
|-------|--------|-------|
| Bbox + tip | `app/telemetry/visualization.py` | State-colored bbox |
| Trajectory | `draw_depth_trajectory` | BGR + thickness ∝ scale |
| Debug HUD | `--debug` | ROIs, LK features, timings |
| JSONL | `app/telemetry/logger.py` | Required fields always |
| Metrics | `app/telemetry/profiler.py` | Latency percentiles, NN rate |

---

## 7. Module map

```text
app/
  main.py, session.py       Application loop
  camera/                   Webcam capture
  ui/roi.py                 ROI selection
  telemetry/                Log, profiler, draw

tracker/
  tracking/
    pipeline.py             Orchestrator
    frame_tracker.py        NN schedule + LK + localizer bridge
    fusion.py               Measurement fusion rules
    peak_trust.py           NN peak EMA gating
    policy/                 FSM, NN scheduler
  backends/pentiptrack.py   NN localizer
  cv/local_flow.py          LK flow
  motion/kalman.py          Kalman 3D
  depth/                    Relative depth
  config.py                 Thresholds
  types.py                  DTOs (PipelineOutput, …)
```

Factory wiring: `tracker/tracking/factory.py` → `create_tracking_pipeline()`.

---

## 8. Performance characteristics

Typical on 720p (hardware-dependent):

- **NN frame:** ~15–50 ms (GPU much faster)
- **Flow-only frame:** ~2–5 ms
- **Session FPS** < **Processing FPS** because MP4 encode + `imshow` + JSONL flush add wall time

Design keeps NN off the hot path most frames (`nn_interval=3` → ~33% NN rate, often higher when peak drops).

---

## 9. Confidence semantics

- **Logged `confidence`:** last NN confidence when NN ran; held between NN frames (not flow confidence)
- **FSM `has_valid_measurement`:** requires **peak-trusted NN**, not flow
- Use **`method_used`** + **`tracking_state`** in logs for post-hoc analysis

---

## 10. Future improvements (given more time)

1. Flow–Kalman gating that rejects flow contradicting prediction (attempted; needs careful tuning)
2. LK-based optical flow scale (radial features, RANSAC, etc.) was meant to help move beyond relative bbox scale, but I did not tune it to stable behavior; asiignment stays relative depth from NN bbox size only for now.
3. Embedded ONNX/TensorRT export for PenTipTrack model.
---

# Failure Cases & Development Lessons

This document covers **observed failure modes** of the current pipeline and **problems encountered during development** — including paths that were tried and reverted. Use it for the assignment’s failure-case section and honest limitation discussion.

---

## 1. Current known failure modes (runtime)

### 1.1 Fast motion & motion blur

**Symptom:** Tip smears; LK features fail or outlier ratio spikes; NN peak drops.

**Why:** LK assumes brightness constancy; blur removes corner texture. NN search window may lag if flow is wrong.

**Behavior:** `uncertain` → `lost`; user re-selects ROI.

**Mitigation today:** Lower `--nn-interval` (NN every 1–2 frames); good lighting; `--debug` to watch red rejected LK points.

---

### 1.2 Finger / hand occlusion

**Symptom:** Bbox still tracks a blob; peak score falls; state → `uncertain` / `lost`.

**Why:** Occluded patch looks unlike template.

**Behavior:** Flow may bridge 1–2 frames if peak stays above `peak_lost_threshold`; otherwise lost.

**Limitation:** No explicit occlusion model — confident green bbox is not drawn when FSM says lost.

---

### 1.3 Cluttered or similar background

**Symptom:** Occasional jump to wrong dark edge or texture.

**Why:** Early prototypes used **template matching** and **pixel Siamese** — both confused by similar patterns. Current NN is better but not perfect.

**Historical note:** This was the main motivation to move from template matching → Siamese NCC → **PenTipTrack** siamese tracker.

---

### 1.4 Pen moves very close to camera

**Symptom:** Scale increases; bbox grows; depth color shifts; sometimes lost if bbox leaves search design.

**Why:** Relative depth is bbox-scale only; extreme scale hits `scale_limits` clamps.

**Behavior:** Trajectory thickens/reddens (by design); if NN peak fails, lost.

**Limitation:** No FOV / distortion model; init ROI size matters.

---

### 1.5 Between NN frames — `uncertain` while still visually OK

**Symptom:** Orange bbox, `method_used=optical_flow`, log shows `tracking_state=uncertain` even when tip is visible.

**Why:** FSM requires **trusted NN** for `visible`. Flow updates Kalman but **does not reset lost streak**.

**Design intent:** Avoid marking “confident” when only fast CV supports the track.

**Tradeoff:** UX looks worse than internal Kalman quality; check `x,y` continuity in log, not only state.

---

### 1.6 Sticky `confidence` in JSONL

**Symptom:** `confidence` stays at last NN value (e.g. `1.0`) on flow frames.

**Why:** Log field is `_last_nn_conf`, not instantaneous measurement quality.

**For analysis:** Filter rows where `method_used` starts with `pentiptrack` or use `--debug` fields (`nn_peak_score`, `nn_inference_used`).

---

### 1.7 Session FPS much lower than Processing FPS

**Symptom:** Console shows Processing FPS ~80, Session FPS ~15.

**Why:** Wall clock includes **MP4 write**, **imshow**, **JSONL flush** — not NN slowness.

**Not a tracking failure** — pipeline latency (`latency_ms` in log) is the right metric for real-time budget.

---

## 2. Problems I hit during development (chronological themes)

### 2.1 Template matching on cluttered desk

**Problem:** NCC template match jumped to keyboard edges and shadows.

**Attempted:** Stricter scores, displacement limits, ambiguous-peak rejection.

**Outcome:** Partial help; abandoned as primary localizer when **PenTipTrack** NN integrated.

**Lesson:** Pixel-level matching is a weak alone localizer for pen tips on busy backgrounds.

---

### 2.2 Siamese NCC (MobileNet) path

**Problem:** Built feature-space NCC tracker (SiamFC-style) as main localizer.

**Issues:** Coordinate mapping bugs, ONNX export friction, still confused on some backgrounds.

**Outcome:** Path **reverted**; production uses in-repo PenTipTrack PyTorch weights instead.

**Lesson:** Siamese idea was sound for assignment; PenTipTrack is a stronger fit for this task.

---

### 2.3 Optical flow “runs away” / accelerates

**Problem:** LK displacement fed into Kalman repeatedly caused overshoot; background features dominated when ROI included clutter.

**Attempts:**

- Flow gating vs Kalman prediction
- Removing `sync_velocity` after flow
- Disabling Kalman update from flow entirely

**Outcome:** Several experiments **broke tracking entirely** (unusable in live sessions) → **hard reset** to a stable commit. Current code uses flow with peak threshold, outlier rejection, and velocity blend — conservative but stable.

**Lesson:** Flow must be gated both by **feature consensus** (MAD outliers) and **peak score**, not raw median alone.

---

### 2.4 Depth / scale from LK features (radial, RANSAC, pairwise ratios)

**Problem:** Pen moved closer to the camera; **scale from LK** lagged, latched to background expansion, or fought display bbox.

**Attempts:** Outer-ring radial scale, RANSAC similarity, inflating LK ROI with accumulated scale, NN anchor blending.

**Outcome:** Adopted **bbox-only scale** for depth — all LK scale paths removed.

**Current:** `RelativeDepthEstimator.observe_bbox(w, h)` on trusted NN frames only.

**Lesson:** Monocular scale from unstructured features is fragile; NN bbox size is slower but interpretable.

---

### 2.5 Lost too aggressive vs too lax

**Problem:** Tuning `nn_interval`, peak thresholds, and bad-peak streak.

**Current coupling:** `_lost_streak > nn_interval` links scheduler and FSM — increasing `--nn-interval` both saves compute **and** tolerates longer NN gaps before lost.

**Instant lost:** single NN frame with `peak < 0.78` after warmup.

---

## 3. How to diagnose from logs

```bash
# Frames where state degraded
grep '"tracking_state": "uncertain"' tracking_log.jsonl | head
grep '"tracking_state": "lost"' tracking_log.jsonl | head

# NN vs bridge methods
grep optical_flow tracking_log.jsonl | wc -l
grep pentiptrack tracking_log.jsonl | wc -l
```

With `--debug` log:

- `nn_peak_score` dropping before lost
- `flow_valid: false` + high `flow_n_points` mismatch → feature problems
- `nn_ms` vs `latency_ms` — NN cost dominates on inference frames

---

## 4. What I would try next

1. **Contradiction gate:** accept flow only if `|flow + pred - pred|` is consistent with recent velocity (failed once — needs softer gate).
2. **Separate confidence** for log/UI on non-NN frames (0 or flow-derived).
3. **Peak-triggered NN** only — already partially via `nn_force_peak`; could be more aggressive when `uncertain`.
4. **Short motion blur detector** → skip LK update, NN-only mode for that frame.

Honest assessment: the pipeline is a **working prototype** suitable for the assignment — not production-ready on all desk/webcam conditions without tuning `nn_interval`.
