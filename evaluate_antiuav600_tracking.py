#!/usr/bin/env python3
"""
Anti-UAV-600 Tracking Evaluation (IR-only)

Evaluates tracker outputs against Anti-UAV-600 ground truth (IR_label.json) and
reports YOLO-style metrics (TP/FP/FN/TN, precision/recall/F1/accuracy).

Dataset format (per sequence):
  dataset_600/<split>/<sequence>/
    - 000001.jpg ...
    - IR_label.json  (contains gt_rect and optional exist)

Predictions format:
  results_600/<sequence>.txt  (x y w h per line, one per frame)
"""
import argparse
import json
from pathlib import Path
import numpy as np


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]."""
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
    if box1[2] <= 0 or box1[3] <= 0 or box2[2] <= 0 or box2[3] <= 0:
        return 0.0

    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_pr_f1_accuracy(predictions, ious, exist_flags, iou_threshold=0.5):
    """Frame-level TP/FP/FN/TN metrics (exist aware)."""
    tp = fp = tn = fn = 0
    for pred, iou, exist in zip(predictions, ious, exist_flags):
        pred_empty = (pred[2] <= 0 or pred[3] <= 0 or
                      (pred[0] == 0 and pred[1] == 0 and pred[2] == 0 and pred[3] == 0))
        if exist > 0:
            if iou >= iou_threshold:
                tp += 1
            else:
                fn += 1
        else:
            if pred_empty:
                tn += 1
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def load_predictions(pred_file):
    predictions = []
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    x, y, w, h = map(float, parts[:4])
                    predictions.append([x, y, w, h])
                except ValueError:
                    predictions.append([0, 0, 0, 0])
            else:
                predictions.append([0, 0, 0, 0])
    return predictions


def load_gt(gt_file):
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    gt_rects = gt_data.get('gt_rect', [])
    exist_flags = gt_data.get('exist')
    if exist_flags is None:
        exist_flags = [1] * len(gt_rects)
    return gt_rects, exist_flags


def evaluate_sequence(seq_dir, pred_file, iou_threshold=0.5):
    gt_file = seq_dir / 'IR_label.json'
    if not gt_file.exists() or not pred_file.exists():
        return None, 'missing_gt_or_pred'

    gt_rects, exist_flags = load_gt(gt_file)
    if len(gt_rects) <= 1:
        return None, 'insufficient_gt'

    predictions = load_predictions(pred_file)

    if len(predictions) != len(gt_rects):
        if len(predictions) < len(gt_rects):
            predictions.extend([[0, 0, 0, 0]] * (len(gt_rects) - len(predictions)))
        else:
            predictions = predictions[:len(gt_rects)]

    ious = [calculate_iou(pred, gt) for pred, gt in zip(predictions, gt_rects)]

    pr_metrics = calculate_pr_f1_accuracy(predictions, ious, exist_flags, iou_threshold=iou_threshold)
    return pr_metrics, None


def main():
    parser = argparse.ArgumentParser(description='Evaluate Anti-UAV-600 tracking results (IR-only).')
    parser.add_argument('--dataset-root', default='./dataset_600', help='Anti-UAV-600 dataset root')
    parser.add_argument('--results-root', default='./results_600', help='Directory with prediction txt files')
    parser.add_argument('--split', default='validation',
                        choices=['train', 'validation', 'track1_test', 'track2_test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for TP matching (default: 0.5)')
    parser.add_argument('--max-seqs', type=int, help='Limit number of sequences')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    results_root = Path(args.results_root)
    split_dir = dataset_root / args.split

    if not dataset_root.exists():
        print(f"ERROR: Dataset root not found: {dataset_root}")
        return
    if not split_dir.exists():
        print(f"ERROR: Split not found: {split_dir}")
        return
    if not results_root.exists():
        print(f"ERROR: Results root not found: {results_root}")
        return

    sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if args.max_seqs:
        sequences = sequences[:args.max_seqs]

    print("=" * 70)
    print("Anti-UAV-600 Tracking Evaluation")
    print("=" * 70)
    print(f"Dataset: {dataset_root}")
    print(f"Split:   {args.split} ({len(sequences)} sequences)")
    print(f"Results: {results_root}")
    print(f"IoU:     {args.iou_threshold}")
    print("=" * 70)

    total_tp = total_fp = total_tn = total_fn = 0
    evaluated = 0
    skipped_missing = 0
    skipped_insufficient = 0

    for seq_dir in sequences:
        seq_name = seq_dir.name
        pred_file = results_root / f"{seq_name}.txt"
        metrics, reason = evaluate_sequence(seq_dir, pred_file, iou_threshold=args.iou_threshold)

        if metrics is None:
            if reason == 'missing_gt_or_pred':
                skipped_missing += 1
                print(f"✗ {seq_name:40s} missing GT or prediction")
            else:
                skipped_insufficient += 1
                print(f"✗ {seq_name:40s} insufficient GT (likely test split)")
            continue

        evaluated += 1
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_tn += metrics['tn']
        total_fn += metrics['fn']
        print(f"✓ {seq_name:40s} evaluated")

    if evaluated == 0:
        print("\nNo sequences evaluated. If using track1_test/track2_test, GT has only init bbox.")
        return

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0

    total_frames = total_tp + total_tn + total_fp + total_fn
    total_gt_frames = total_tp + total_fn

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Evaluated sequences: {evaluated}/{len(sequences)}")
    if skipped_missing:
        print(f"Skipped (missing GT/pred): {skipped_missing}")
    if skipped_insufficient:
        print(f"Skipped (insufficient GT): {skipped_insufficient}")

    print("\nDetection Metrics (IoU >= threshold, exist flag aware)")
    print(f"   Total frames: {total_frames}")
    print(f"   Total GT frames (exist=1): {total_gt_frames}")
    print(f"   True Positives:  {total_tp}")
    print(f"   False Positives: {total_fp}")
    print(f"   False Negatives: {total_fn}")
    print(f"   True Negatives:  {total_tn}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1_score:.4f}")
    print(f"   Accuracy:  {accuracy:.4f}")

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
