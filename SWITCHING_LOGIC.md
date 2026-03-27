# Anti-UAV Switching Logic (Current Codebase)

This document explains how switching between detection and tracking happens in the current Anti-UAV pipeline, with pointers to the exact files and functions.

## Where the switch happens (fixed schedule)

**File:** `Anti-UAV/Codes/batch_track_anti_uav_rgbt.py`  
**Function:** `track_video(...)`

The logic is time‑based (not confidence‑based). Detection runs every 50 frames, tracking runs in between.

Key line:
- `if frame_count % 50 == 1:`  
  This is the **50‑frame switch** condition.

Inside this branch:
- **Detection head** is called via:
  - `init_box = detector.forward_IR(frame)` (or `forward_RGB` depending on modality)
- If a box is returned:
  - First detection: `tracker.init_track(init_box, frame)`
  - Subsequent detections: `tracker.change_state(init_box)`

Outside this branch (the `else` path):
- **Tracking head** is called via:
  - `outputs = tracker.on_track(frame)`

## Scenario‑by‑scenario behavior

### 1) First visible frame
**Head used:** Detection head
- On the first detection frame, detector returns a box.
- Tracker is initialized with `tracker.init_track(init_box, frame)`.

### 2) After first visible, while target remains visible
**Head used:** Tracking head
- For the next 49 frames, tracking runs using `tracker.on_track(frame)`.
- The tracker updates its internal template each frame.

### 3) Every 50th frame (scheduled re‑detection)
**Head used:** Detection head
- Detector runs again and, if it returns a box, the tracker is re‑centered using `tracker.change_state(init_box)`.

### 4) After first visible, but target becomes NOT visible
**Head used:** Tracking head (still)
- There is **no visibility check** in the current code.
- The tracker continues to run even if the target is occluded.
- If tracking fails, it reuses the last prediction (see below).

### 5) Detector fails to return a box
**Head used:** Tracking head (or last box)
- If detection returns no box, the code falls back to the last prediction:
  - `predictions.append(predictions[-1])` (if any), else `[0,0,0,0]`.

## Tracker internals (where tracking output comes from)

**File:** `Anti-UAV/Codes/tracking_wrapper/dronetracker/trackinguav/evaluation/tracker.py`  
**Class:** `Tracker`  
**Method:** `on_track(self, frame)`

This method returns only the bounding box:
- `boundingbox = self.tracker.update(frame)`
- `return boundingbox`

## Detector internals (where detection output comes from)

**File:** `Anti-UAV/Codes/detect_wrapper/Detectoruav.py`  
**Methods:** `forward_IR(...)`, `forward_RGB(...)`

These methods run YOLO inference and return **only the best bounding box** (confidence is not returned in current code).

## Key takeaway

The pipeline **does not use adaptive confidence switching**. It is purely **time‑based**:
- Detect every 50 frames
- Track in between
- No visibility or confidence check is applied in the switching logic

If you want confidence‑based switching (as in the paper figure), you’d need to modify both the detector and tracker to return confidence values and update the decision logic accordingly.
