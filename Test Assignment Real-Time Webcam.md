# Test Assignment: Real-Time Webcam Pen-Tip Tracking with Hybrid ML + Classical CV

## Context

We are developing a (relatively) low-power device that needs immediate object localization from a video stream. The final product has strong real-time constraints, so we do not expect a heavy neural network to process every frame.

Instead, we expect a hybrid perception pipeline:

* Neural network for robust detection / correction
* Classical computer vision for fast frame-to-frame tracking
* Kalman filtering for motion prediction, smoothing, and short-term loss handling
* A proper evaluation harness to measure latency, tracking quality, and failure cases

We are also comfortable with AI-assisted development. You are expected to use either **Claude Code** or **Codex** during the assignment. We care less about whether every line of code is handwritten and more about your ability to design the system, use agents responsibly, test the result, and explain the engineering process.

---

# Objective

Build a real-time webcam application that tracks the **tip of a pen** while the user moves it in front of the camera.

Example scenarios:

* The user writes something on paper.
* The user moves the pen in the air.
* The pen moves closer to and farther from the camera.
* The pen is partially occluded by fingers or motion blur.
* The background is moderately cluttered.

The application should produce a recorded video where the predicted pen-tip trajectory is drawn on top of the webcam image.

The trajectory should represent a **3D path**:

* `x` and `y` are the image-plane coordinates of the pen tip.
* `z` is estimated relative depth.
* Since this is a monocular webcam task, we do not expect perfect metric depth unless you decide to add calibration. Approximate / perceived / relative depth is acceptable, but the method must be explained.

The depth should be visualized in the output video using:

* Line color, or
* Line thickness, or
* Both

For example:

* Thin/dark line = farther away
* Thick/bright line = closer to camera

---

# Core Requirements

## 1. Live webcam input

The system must read from a webcam in real time.

It should display a live preview with the detected pen tip and trajectory overlay.

The final result must include a recorded video with the overlay.

Post-processing is allowed for analysis and reporting, but the main tracking output should be produced by the real-time pipeline, not reconstructed offline afterward.

---

## 2. Real-time tracking

The system must output the pen-tip location for every processed frame.

The application should report:

* Actual camera FPS
* Processing FPS
* Average latency per frame
* 95th percentile latency
* 99th percentile latency
* Number of dropped or skipped frames
* Neural-network inference rate

Latency is more important than just FPS.

For example, a system that runs at 30 FPS but delays the video by 500 ms is not good enough for our use case.

---

## 3. Required algorithmic components

Your solution must include all three of the following:

### A. Neural network component

Use a neural network for at least one of these:

* Pen / pen-tip detection
* Object segmentation
* Feature extraction
* Local correction when the classical tracker drifts
* Depth estimation
* Re-identification after temporary loss

You may use a pretrained model, fine-tune a model, train a small custom model, or use a lightweight local model. Cloud inference is not allowed.

The neural network does not need to run on every frame.

A good design may run the neural network periodically or only when confidence drops.

---

### B. Classical computer vision component

Use classical CV methods for fast tracking between neural-network outputs.

Examples:

* Optical flow
* Template matching
* Keypoint tracking
* Contour tracking
* Color / shape-based tracking
* OpenCV tracker
* Frame differencing
* Local search around the predicted position

The classical CV part should help provide immediate frame-by-frame position estimates without waiting for the neural network every time.

---

### C. Kalman filter

Use a Kalman filter or a clearly equivalent state-space estimator.

The Kalman filter should help with at least some of the following:

* Predicting the pen-tip position in the next frame
* Smoothing noisy detections
* Handling short detection gaps
* Estimating velocity
* Estimating relative depth movement
* Combining measurements from NN and classical CV

The report should explain the state vector you used.

Example:

```text
[x, y, z, vx, vy, vz]
```

Where:

* `x, y` = image position
* `z` = estimated relative depth
* `vx, vy, vz` = estimated motion speed

---

# Expected System Architecture

We expect something similar to this type of architecture:

```text
Webcam frame
    ↓
Frame timestamping
    ↓
Fast prediction from Kalman filter
    ↓
Classical CV tracker searches near predicted position
    ↓
Neural network runs periodically or when confidence is low
    ↓
Measurement fusion: NN + CV + Kalman
    ↓
Confidence estimation
    ↓
3D trajectory update
    ↓
Live visualization
    ↓
Recorded output video + logs
```

The exact implementation is up to you, but the architecture must be explained clearly.

---

# Initialization

You may choose the initialization method.

Acceptable options:

* User clicks on the pen tip in the first frame.
* User draws a bounding box around the pen tip.
* System automatically detects the pen tip.
* System asks the user to hold the pen still for the first second.
* System uses a colored marker or visible pen tip, as long as the full solution is not only a simple color threshold.

You should explain the limitations of your initialization method.

---

# 3D / Depth Requirement

The output should include a 3D trajectory estimate.

With a single webcam, true metric depth is hard. We understand this. You may estimate relative depth using one or more of the following:

* Apparent pen size
* Tip sharpness / blur
* Motion scale
* Monocular depth model
* Known pen dimensions
* Camera calibration
* User-provided calibration movement
* Assumption that the pen moves on a plane
* Any other reasonable method

The important part is not perfect depth accuracy. The important part is that you:

1. Produce a reasonable `z` estimate.
2. Explain how it is calculated.
3. Visualize it clearly.
4. Describe when and why it fails.

For every frame, save something like:

```json
{
  "frame": 127,
  "timestamp_ms": 4233.2,
  "x": 318.4,
  "y": 241.9,
  "z_relative": 0.63,
  "confidence": 0.82,
  "tracking_state": "visible",
  "method_used": "cv_kalman",
  "latency_ms": 18.7
}
```

Possible `tracking_state` values:

```text
visible
uncertain
lost
reacquired
```

Possible `method_used` values:

```text
neural_network
optical_flow
template_matching
keypoint_tracking
kalman_prediction
mixed
```

---

# Output Deliverables

Please submit the following.

## 1. Git repository

Submit a Git repository with the full development history.

Do not squash everything into one final commit.

We want to see the engineering process.

The commit history should show the story of how the solution evolved.

Example commit structure:

```text
01 - Initial project setup
02 - Add webcam capture and recording
03 - Add manual pen-tip initialization
04 - Add baseline OpenCV tracker
05 - Add Kalman filter prediction
06 - Add neural-network detector / correction step
07 - Add trajectory visualization
08 - Add relative depth estimation
09 - Add latency and FPS logging
10 - Add benchmark harness
11 - Add failure-case recordings
12 - Refactor pipeline architecture
13 - Final report and README
```

Commit messages should be meaningful.

A huge single commit with the full solution is a negative signal.

---

## 2. Runnable application

The application should run locally.

Please include:

* Setup instructions
* Dependency list
* How to run the webcam demo
* How to choose the camera
* How to initialize the target
* How to save the output video
* How to save logs

Example command:

```bash
python run_webcam_tracker.py --camera 0 --output demo_output.mp4 --log tracking_log.jsonl
```

---

## 3. Recorded demo videos

Submit at least three recorded demo videos:

### Demo 1 — Simple movement

Pen tip moves slowly and clearly in front of the camera.

### Demo 2 — Writing or drawing

User writes something on paper or draws a simple shape.

### Demo 3 — Harder case

Include at least one difficulty:

* Faster motion
* Motion blur
* Partial occlusion by fingers
* Cluttered background
* Moving closer/farther from camera
* Temporary loss and re-acquisition

Each output video should show:

* Original webcam frame
* Current predicted pen-tip position
* Trajectory line
* Depth indication using line color or thickness
* Optional confidence display
* Optional tracking state display

---

## 4. Per-frame tracking log

Save a per-frame log in JSONL or CSV format.

Each frame should include:

```text
frame index
timestamp
x
y
z_relative
confidence
tracking_state
method_used
latency_ms
```

Bonus if you also include:

```text
NN inference used: yes/no
NN inference time
CV tracking time
Kalman update time
total pipeline time
```

---

## 5. Architecture report

Submit a short report, around 3–6 pages.

It should explain:

* Overall pipeline architecture
* Why you chose this architecture
* What the neural network does
* What the classical CV part does
* How the Kalman filter is used
* How measurements are combined
* How confidence is calculated
* How target loss is detected
* How re-acquisition works
* How relative depth is estimated
* How the trajectory is visualized
* Main failure cases
* What you would improve with more time

We are especially interested in your reasoning and tradeoffs.

---

## 6. Methodology and AI-agent report

You must use either **Claude Code** or **Codex** during the assignment.

Include a short section describing:

* Which agent you used
* What tasks you delegated to the agent
* How you structured the work
* Whether you used one agent or multiple agent sessions
* What the agent got wrong
* How you verified generated code
* How you tested edge cases
* How you prevented yourself from blindly trusting generated output

We do not penalize AI usage. We encourage it.

However, you must be able to explain and defend the final solution.

A good answer is not:

```text
I asked an agent to build the whole thing.
```

A good answer is:

```text
I used an agent to generate a baseline OpenCV tracker, another session to review the Kalman implementation, and another to suggest latency metrics. I verified the result using recorded test cases, per-frame logs, and manual inspection of failure cases.
```

---

# Evaluation Metrics

Your report should include measured results.

At minimum, include:

## Real-time performance

```text
Average FPS
Average latency
95th percentile latency
99th percentile latency
Dropped frames
NN inference frequency
```

## Tracking quality

```text
Does the trajectory visually follow the pen tip?
How often does tracking become unstable?
How often does the system lose the target?
How quickly does it recover?
How much jitter is visible?
```

## Trajectory smoothness

Explain whether the Kalman filter improves or worsens the result.

For example:

```text
Without Kalman: noisy but responsive
With Kalman: smoother but slightly delayed
```

## Failure cases

Show and explain at least three failure cases.

Examples:

```text
Fast pen movement causes motion blur.
Finger occludes the tip.
Background has a similar dark object.
Pen moves too close to the camera.
Depth estimate becomes unstable.
```

---

# Technical Constraints

* Must run from a live webcam.
* Must work locally.
* Cloud inference is not allowed.
* Must include neural network + classical CV + Kalman filter.
* Must output target position for every processed frame.
* Must record output video with trajectory overlay.
* Must save per-frame tracking data.
* Must include meaningful Git commit history.
* Must use Claude Code or Codex and explain how it was used.

You may use open-source libraries such as:

```text
OpenCV
PyTorch
TensorFlow
ONNX Runtime
NumPy
SciPy
MediaPipe
Ultralytics
scikit-image
filterpy
```

These are examples only. You may choose your own stack.

---

# What We Care About Most

We are not primarily testing whether you can manually write every line of tracking code.

We are testing whether you can:

* Design a real-time perception pipeline
* Combine neural networks with classical computer vision
* Use Kalman filtering correctly
* Build a useful test and evaluation harness
* Measure latency and robustness
* Understand failure modes
* Use AI agents effectively
* Keep a clean engineering history in Git
* Explain your architecture and tradeoffs clearly

---

# Suggested Time Budget

Please spend around **6–10 hours**.

We do not expect a perfect production-level result.

We expect a working prototype, a clear architecture, measured results, and honest discussion of limitations.

---

# Evaluation Criteria

## 1. Architecture and reasoning — 25%

Strong candidates should explain why the pipeline is structured the way it is.

We want to see clear separation between:

```text
Neural-network detection/correction
Fast classical CV tracking
Kalman prediction/smoothing
Depth estimation
Visualization
Logging
Evaluation
```

---

## 2. Real-time behavior — 20%

The system should be designed for immediate output.

Good signs:

* Low frame latency
* No unnecessary frame buffering
* Every frame gets a predicted position or a clear lost/uncertain state
* NN does not block the whole pipeline unnecessarily
* Latency is measured, not guessed

---

## 3. Hybrid tracking implementation — 20%

The solution must use NN + classical CV + Kalman in a meaningful way.

A weak solution would simply run a detector on every frame.

A better solution would use the NN for detection/correction and classical CV + Kalman for fast intermediate tracking.

---

## 4. Test harness and metrics — 15%

We value the ability to test and improve the system.

Good signs:

* Per-frame logs
* Latency measurements
* Demo recordings
* Failure-case analysis
* Comparison of different pipeline settings
* Clear run instructions

---

## 5. Git history and AI-agent workflow — 15%

We want to see how the work was done.

Good signs:

* Small meaningful commits
* Clear commit messages
* Visible development progression
* Agent usage explained
* Generated code verified
* Mistakes and corrections documented

---

## 6. Practical robustness — 5%

Good signs:

* Handles temporary occlusion
* Has an uncertainty/lost state
* Does not confidently draw nonsense when the pen is lost
* Has reasonable behavior under blur or clutter

---

# Red Flags

The following are negative signals:

* One huge final commit
* No explanation of architecture
* No latency measurement
* No per-frame logs
* Pure neural-network solution only
* Pure OpenCV solution only
* Kalman filter included but not actually useful
* No live webcam support
* Output generated only by offline post-processing
* No failure-case analysis

---

# Final Submission Checklist

Please submit:

```text
[ ] Git repository with meaningful commit history
[ ] README with setup and run instructions
[ ] Live webcam tracking application
[ ] Recorded output videos
[ ] Per-frame JSONL or CSV logs
[ ] Architecture report
[ ] Metrics / benchmark report
[ ] AI-agent usage report
[ ] Explanation of failure cases
```

The final solution does not need to be perfect. We are mainly interested in how you approach a real-time perception problem, how you validate your work, and how you use modern AI-assisted engineering tools responsibly.
