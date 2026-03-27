# Anti-UAV top-level scripts overview

This file summarizes the standalone Python scripts in the `Anti-UAV/` folder that can be run directly (no package entrypoint). Most of them wrap the detector/tracker in `Anti-UAV/Codes/` and/or evaluate saved result files.

## Evaluation scripts

### `evaluate_anti_uav_rgbt.py`
- **Purpose:** Evaluate Anti-UAV-300 tracking outputs (RGBT/visible/infrared) using IoU and `exist` flags.
- **Inputs:** `--dataset-root` (default `./dataset`), `--results-root` (default `./results`), `--subset` (`train|val|test`), `--modality` (`infrared|visible`), `--iou`.
- **Outputs:** Prints TP/FP/FN/TN and overall precision/recall/F1/accuracy (plus per-video stats).
- **Example:** `python evaluate_anti_uav_rgbt.py --subset test --modality infrared`

### `evaluate_antiuav600_tracking.py`
- **Purpose:** Evaluate Anti-UAV-600 tracking outputs against `IR_label.json` for train/val subsets.
- **Inputs:** `--dataset-root` (default `./dataset_600`), `--results-root` (default `./results_600`), `--subset` (`train|val|track1_test|track2_test`), `--iou`.
- **Outputs:** Prints TP/FP/FN/TN and precision/recall/F1/accuracy; warns that test subsets only have init bboxes.
- **Example:** `python evaluate_antiuav600_tracking.py --subset train`

### `evaluate_antiuav600_coco.py`
- **Purpose:** Evaluate Anti-UAV-600 **COCO-format** detection dataset using EDTC by reconstructing sequences from CSVs.
- **Inputs:** `--dataset-root`, `--csv-file`, `--iou` (via CLI args in the script).
- **Outputs:** EDTC per sequence and overall (printed to stdout).
- **Example:** `python evaluate_antiuav600_coco.py --dataset-root ./dataset_600 --csv-file train.csv`

## Tracking scripts (videos)
### `track_generic_dataset.py`
- **Purpose:** Track any Anti-UAV-format dataset stored as videos; reports simple timing/FPS stats.
- **Inputs:** `--dataset-root`, `--results-root`, `--subset`, `--modality`, `--video-ext`, `--max-videos`.
- **Outputs:** Results files in the target results directory; prints tracking/detection FPS.
- **Example:** `python track_generic_dataset.py --dataset-root /path/to/dataset --subset test --modality infrared`

## Tracking scripts (frame sequences)

### `track_frame_sequences.py`
- **Purpose:** Track frame-sequence datasets (Anti-UAV-600 or similar) with argparse support.
- **Inputs:** `--dataset-root` (required), `--results-root`, `--subset`, `--modality` (`infrared|visible`), `--max-videos`, `--start-index`, `--gpu-id`.
- **Outputs:** Per-sequence results in `results_600/` (or custom).
- **Example:** `python track_frame_sequences.py --dataset-root /path/to/3rd_Anti-UAV_train_val --subset track1_test --modality infrared`

### `run_tracking_600_optimized.py`
- **Purpose:** Optimized Anti-UAV-600 tracking loop with memory cleanup and periodic progress logs.
- **Inputs:** Positional args: `[subset] [max_videos]`.
- **Outputs:** `results_600/<sequence>.txt`.
- **Example:** `python run_tracking_600_optimized.py track1_test 10`

## Visualization

### `visualize_antiuav300.py`
- **Purpose:** Render Anti-UAV-300 videos with GT (green) and prediction (blue) boxes.
- **Inputs:** `--dataset-root`, `--results-root`, `--subset`, `--modality`, `--video-name` (single) or `--all` (batch), `--output`/`--output-dir`.
- **Outputs:** Rendered MP4s in the target output path/dir.
- **Example:** `python visualize_antiuav300.py --subset test --modality infrared --all --output-dir ./visualizations`

