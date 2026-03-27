#!/usr/bin/env python3
"""
Visualize Anti-UAV-300 tracking outputs with GT + predictions.

- Green: Ground-truth bounding box (gt_rect)
- Blue: Prediction bounding box (results file)

Dataset layout (per video):
  Anti-UAV/dataset/<subset>/<video_name>/
    - infrared.json / visible.json
    - infrared.mp4 / visible.mp4

Predictions layout:
  Anti-UAV/results/<video_name>.txt   (x y w h per frame)
"""
import argparse
import json
from pathlib import Path
import cv2


def load_gt(gt_path):
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    gt_rects = gt_data.get('gt_rect', [])
    exist_flags = gt_data.get('exist')
    if exist_flags is None:
        exist_flags = [1] * len(gt_rects)
    return gt_rects, exist_flags


def load_predictions(pred_path):
    preds = []
    with open(pred_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    x, y, w, h = map(float, parts[:4])
                    preds.append([x, y, w, h])
                except ValueError:
                    preds.append([0, 0, 0, 0])
            else:
                preds.append([0, 0, 0, 0])
    return preds


def draw_box(frame, box, color, label=None):
    if box is None:
        return
    x, y, w, h = [int(v) for v in box]
    if w <= 0 or h <= 0:
        return
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(frame, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    parser = argparse.ArgumentParser(description='Visualize Anti-UAV-300 GT + tracker outputs')
    parser.add_argument('--dataset-root', default='./dataset', help='Anti-UAV-300 dataset root')
    parser.add_argument('--results-root', default='./results', help='Results directory with .txt files')
    parser.add_argument('--subset', default='test', choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--video-name', help='Video folder name (e.g. 20190925_111757_1_1)')
    parser.add_argument('--all', action='store_true', help='Process all videos in the split')
    parser.add_argument('--modality', default='infrared', choices=['infrared', 'visible'], help='Video modality')
    parser.add_argument('--output', default='./output_preview.mp4', help='Output video file (single-video mode)')
    parser.add_argument('--output-dir', default='./visualizations', help='Output directory (all-videos mode)')
    parser.add_argument('--max-frames', type=int, help='Optional cap on number of frames')
    parser.add_argument('--show', action='store_true', help='Preview frames in a window while writing')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output videos')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    results_root = Path(args.results_root)
    split_dir = dataset_root / args.subset

    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    if args.all:
        video_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not args.video_name:
            raise ValueError("--video-name is required unless --all is set")
        video_names = [args.video_name]
        output_dir = None

    for video_name in video_names:
        video_dir = split_dir / video_name
        gt_path = video_dir / f"{args.modality}.json"
        video_path = video_dir / f"{args.modality}.mp4"
        pred_path = results_root / f"{video_name}.txt"

        if not gt_path.exists() or not video_path.exists() or not pred_path.exists():
            print(f"Skipping {video_name}: missing GT, video, or prediction")
            continue

        gt_rects, exist_flags = load_gt(gt_path)
        preds = load_predictions(pred_path)

        if len(preds) < len(gt_rects):
            preds.extend([[0, 0, 0, 0]] * (len(gt_rects) - len(preds)))
        elif len(preds) > len(gt_rects):
            preds = preds[:len(gt_rects)]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skipping {video_name}: could not open video")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_dir:
            output_path = output_dir / f"{video_name}_{args.modality}.mp4"
        else:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            print(f"Skipping {video_name}: output exists ({output_path})")
            cap.release()
            continue

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < len(gt_rects):
                gt_box = gt_rects[frame_idx]
                pred_box = preds[frame_idx]
                exist = exist_flags[frame_idx] if frame_idx < len(exist_flags) else 1

                if exist > 0:
                    draw_box(frame, gt_box, (0, 255, 0), label='GT')
                draw_box(frame, pred_box, (255, 0, 0), label='Pred')

            writer.write(frame)

            if args.show:
                cv2.imshow('Anti-UAV-300 Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

        print(f"Saved visualization to: {output_path}")


if __name__ == '__main__':
    main()
