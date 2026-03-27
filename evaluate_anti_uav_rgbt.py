#!/usr/bin/env python3
"""
Anti-UAV-RGBT Dataset Evaluation Tool
======================================

Evaluate tracking results on Anti-UAV-RGBT dataset (RGB + Thermal)

Dataset Format:
  /Anti-UAV-RGBT/test/video_name/
    ├── infrared.json (GT with gt_rect and exist)
    ├── infrared.mp4
    ├── visible.json
    └── visible.mp4

Prediction Format:
  results/video_name.txt (x y w h, one per line)

Usage:
  python evaluate_anti_uav_rgbt.py \
      --dataset-root /home/varad/Downloads/Anti-UAV-RGBT \
      --results-root ./my_tracking_results \
      --subset test
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
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


def calculate_edtc(predictions, ious, exist_flags, alpha=0.2, beta=0.3):
    """
    Paper's Formula: Acc = Σ[IoU_t × δ(v_t > 0) + p_t × (1 - δ(v_t > 0))] / T 
                           - α × (Σ q_t × δ(v_t > 0) / T*)^β
    
    CORRECTED logic:
    
    p_t (predicted visibility flag):
        - When target is NOT visible (v_t = 0):
          - p_t = 1 if predicted box is EMPTY (all zeros)
          - p_t = 0 if predicted box exists (non-zero)
    
    q_t (detection failure flag):
        - When target IS visible (v_t > 0):
          - q_t = 1 if IoU = 0.0 (predicted empty when target exists)
          - q_t = 0 if IoU > 0.0 (caught some target region)
    """
    
    if not ious or not exist_flags or not predictions:
        return 0.0
    
    T = len(exist_flags)  # Total frames
    T_star = sum(1 for v in exist_flags if v > 0)  # Frames where target visible
    
    if T_star == 0:
        return 0.0
    
    # Calculate T1 (Reward term)
    T1 = 0.0
    for t in range(T):
        if exist_flags[t] > 0:  # Target IS visible
            T1 += ious[t]
        else:  # Target is NOT visible
            # p_t: 1 if predicted box is empty, 0 if predicted box exists
            pred_box = predictions[t]
            is_pred_empty = (pred_box[2] <= 0 or pred_box[3] <= 0 or 
                           (pred_box[0] == 0 and pred_box[1] == 0 and 
                            pred_box[2] == 0 and pred_box[3] == 0))
            p_t = 1.0 if is_pred_empty else 0.0
            T1 += p_t
    
    reward_term = T1 / T
    
    # Calculate T2 (Penalty term)
    T2 = 0.0
    for t in range(T):
        if exist_flags[t] > 0:  # Target IS visible
            # q_t: 1 if IoU = 0.0 (predicted empty when target exists), 0 otherwise
            q_t = 1.0 if ious[t] == 0.0 else 0.0
            T2 += q_t
        else:  # Target is NOT visible
            T2 += 0.0  # No penalty when target doesn't exist
    
    penalty_rate = T2 / T_star
    penalty_term = alpha * (penalty_rate ** beta)
    
    # Final EDTC score
    edtc = reward_term - penalty_term
    
    return edtc


def calculate_success_rate(ious, exist_flags, threshold=0.5):
    """Calculate success rate: % of frames with IoU >= threshold"""
    valid_ious = [iou for iou, exist in zip(ious, exist_flags) if exist == 1]
    
    if not valid_ious:
        return 0.0
    
    successes = sum(1 for iou in valid_ious if iou >= threshold)
    return (successes / len(valid_ious)) * 100


def calculate_pr_f1_accuracy(predictions, ious, exist_flags, iou_threshold=0.5):
    """
    Compute precision/recall/F1/accuracy on tracking outputs.

    Frame-level logic:
      - If target exists (exist=1) and IoU >= threshold => TP
      - If target exists (exist=1) and IoU < threshold  => FN
      - If target not exists (exist=0) and pred empty  => TN
      - If target not exists (exist=0) and pred box    => FP
    """
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


def evaluate_video(dataset_root, results_root, video_name, modality='infrared', alpha=0.2, beta=0.3):
    """Evaluate a single video"""
    
    # Load ground truth
    gt_file = Path(dataset_root) / video_name / f'{modality}.json'
    if not gt_file.exists():
        return None
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    gt_rects = gt_data.get('gt_rect', [])
    exist_flags = gt_data.get('exist', [])
    
    # Load predictions
    pred_file = Path(results_root) / f'{video_name}.txt'
    if not pred_file.exists():
        return None
    
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
    
    # Ensure same length
    if len(predictions) != len(gt_rects):
        if len(predictions) < len(gt_rects):
            predictions.extend([[0, 0, 0, 0]] * (len(gt_rects) - len(predictions)))
        else:
            predictions = predictions[:len(gt_rects)]
    
    # Validate all predictions have 4 values
    for i, pred in enumerate(predictions):
        if len(pred) < 4:
            predictions[i] = [0, 0, 0, 0]
    
    # Calculate IoU for each frame
    ious = []
    for pred, gt in zip(predictions, gt_rects):
        iou = calculate_iou(pred, gt)
        ious.append(iou)
    
    # Calculate metrics - NOW PASS predictions list
    edtc = calculate_edtc(predictions, ious, exist_flags, alpha=alpha, beta=beta)

    pr_metrics = calculate_pr_f1_accuracy(predictions, ious, exist_flags, iou_threshold=0.5)
    
    success_rates = {}
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        success_rates[threshold] = calculate_success_rate(ious, exist_flags, threshold)
    
    return {
        'edtc': edtc,
        'success_rates': success_rates,
        'num_frames': len(gt_rects),
        'ious': ious,
        'pr_metrics': pr_metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate tracking results on Anti-UAV-RGBT dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Evaluate test set
  python evaluate_anti_uav_rgbt.py \\
      --dataset-root /home/varad/Downloads/Anti-UAV-RGBT \\
      --results-root ./my_tracking_results \\
      --subset test

  # Evaluate with infrared modality
  python evaluate_anti_uav_rgbt.py \\
      --dataset-root /home/varad/Downloads/Anti-UAV-RGBT \\
      --results-root ./my_tracking_results \\
      --subset test \\
      --modality infrared
        '''
    )
    
    parser.add_argument('--dataset-root', default='./dataset', help='Anti-UAV-RGBT dataset root')
    parser.add_argument('--results-root', default='./results', help='Directory with prediction txt files')
    parser.add_argument('--subset', default='test', choices=['train', 'val', 'test'],
                       help='Dataset subset to evaluate')
    parser.add_argument('--modality', default='infrared', choices=['infrared', 'visible'],
                       help='Which modality to evaluate (infrared or visible)')
    parser.add_argument('--max-videos', type=int, help='Limit number of videos')
    parser.add_argument('--alpha', type=float, default=0.2, help='Penalty weight (default: 0.2)')
    parser.add_argument('--beta', type=float, default=0.3, help='Penalty exponent (default: 0.3)')
    
    args = parser.parse_args()
    
    # Validate paths
    dataset_root = Path(args.dataset_root)
    results_root = Path(args.results_root)
    subset_dir = dataset_root / args.subset
    
    if not dataset_root.exists():
        print(f"ERROR: Dataset root not found: {args.dataset_root}")
        return
    
    if not subset_dir.exists():
        print(f"ERROR: Subset not found: {subset_dir}")
        return
    
    if not results_root.exists():
        print(f"ERROR: Results root not found: {args.results_root}")
        return
    
    # Get all videos in subset
    videos = sorted([d.name for d in subset_dir.iterdir() if d.is_dir()])
    if args.max_videos:
        videos = videos[:args.max_videos]
    
    print("=" * 70)
    print(f"Anti-UAV-RGBT EVALUATION")
    print("=" * 70)
    print(f"Dataset: {args.dataset_root}")
    print(f"Results: {args.results_root}")
    print(f"Subset: {args.subset} ({len(videos)} videos)")
    print(f"Modality: {args.modality}")
    print(f"Parameters: alpha={args.alpha}, beta={args.beta}")
    print("=" * 70)
    print()
    
    # Evaluate each video
    results = {}
    valid_count = 0
    
    for video_name in videos:
        result = evaluate_video(dataset_root / args.subset, results_root, 
                               video_name, modality=args.modality, 
                               alpha=args.alpha, beta=args.beta)
        
        if result is not None:
            results[video_name] = result
            valid_count += 1
            print(f"✓ {video_name:40s} evaluated")
        else:
            print(f"✗ {video_name:40s} FAILED")
    
    if not results:
        print("ERROR: No videos evaluated successfully")
        return
    
    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    
    # Calculate overall metrics (micro-averaged across frames)
    total_tp = sum(r['pr_metrics']['tp'] for r in results.values())
    total_fp = sum(r['pr_metrics']['fp'] for r in results.values())
    total_tn = sum(r['pr_metrics']['tn'] for r in results.values())
    total_fn = sum(r['pr_metrics']['fn'] for r in results.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0

    total_frames = total_tp + total_tn + total_fp + total_fn
    total_gt_frames = total_tp + total_fn

    print(f"\nDetection Metrics (IoU >= 0.5, exist flag aware)")
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