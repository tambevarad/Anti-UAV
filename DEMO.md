# Anti-UAV Tracking Demo

Real-time UAV tracking using RGB-Thermal (RGBT) fusion with DroneDetection and KCF-style tracking.

## Project Structure

```
Anti-UAV/
├── Codes/                          # Core implementation
│   ├── detect_wrapper/            # Detection module (YOLOv5)
│   ├── tracking_wrapper/          # Tracking module (KCF)
│   └── demo_detect_track_fixed.py # Main demo script
├── anti_uav_jittor/               # Jittor implementations (reference)
├── dataset/                        # Anti-UAV-RGBT dataset
│   ├── test/                      # 91 test videos
│   ├── train/                     # Training videos
│   └── val/                       # Validation videos
├── results/                        # Tracking results (predictions)
├── evaluate_anti_uav_rgbt.py      # Evaluation script (precision/recall/F1)
└── README.md                       # Main documentation
```

## Quick Start

### 1. Run Demo on Sample Video

```bash
cd Codes
python demo_detect_track_fixed.py
```

This runs tracking on a sample video with visualization.

### 2. Evaluate on Anti-UAV-RGBT Dataset

```bash

# Activate venv
source venv/bin/activate

# Evaluate all test videos
python evaluate_anti_uav_rgbt.py

# Or with custom paths
python evaluate_anti_uav_rgbt.py \
    --dataset-root ./dataset \
    --results-root ./results \
    --subset test \
    --modality infrared
```

## Dataset Format

The Anti-UAV-RGBT dataset structure:

```
dataset/test/video_name/
├── infrared.json      # GT annotations (thermal modality)
├── infrared.mp4       # Thermal video
├── visible.json       # GT annotations (RGB modality)
└── visible.mp4        # RGB video
```

Each JSON contains:
- `gt_rect`: List of [x, y, w, h] bounding boxes
- `exist`: List of 0/1 indicating frame validity

## Expected Results

`evaluate_anti_uav_rgbt.py` prints per-video and overall metrics:

- True/False Positives/Negatives (TP/FP/FN/TN)
- Precision, Recall, F1, Accuracy

## Key Files

- `demo_detect_track_fixed.py` - Run tracking on videos
- `evaluate_anti_uav_rgbt.py` - Evaluate predictions (precision/recall/F1)
- Prediction format: `results/video_name.txt` (x y w h, one per line)

## Dependencies

- Python 3.8+
- PyTorch
- OpenCV
- Numba
- YOLOv5

See `venv/` for installed packages.
