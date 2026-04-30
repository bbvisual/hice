"""
Evaluate pre-computed predictions from a JSON file.

Usage:
    python eval_from_json.py <pred_json> <gt_json> [--tolerances 0 1 2 4 8] [--windows 1 3]

The prediction JSON must follow the same format as the files saved by eval.py:
    [{"video": "...", "events": [{"label": "...", "frame": <int>, "score": <float>}, ...]}, ...]

The ground-truth JSON is the standard dataset label file:
    [{"video": "...", "events": [{"frame": <int>, "label": "..."}], ...}, ...]
"""

import argparse
import json
import sys
import os

import numpy as np

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(__file__))

from util.eval import non_maximum_supression, soft_non_maximum_supression
from util.score import compute_mAPs

TOLERANCES = [0, 1, 2]
WINDOWS = [1, 3]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def align_predictions_to_gt(pred, gt):
    """
    Ensure pred and gt cover exactly the same set of videos.
    Videos in gt but missing from pred get an empty events list.
    Videos in pred but not in gt are dropped (with a warning).
    """
    gt_videos = {x['video'] for x in gt}
    pred_videos = {x['video'] for x in pred}

    extra = pred_videos - gt_videos
    if extra:
        print(f'[WARNING] {len(extra)} video(s) in predictions not found in ground truth — skipping them.')
        pred = [x for x in pred if x['video'] in gt_videos]

    missing = gt_videos - {x['video'] for x in pred}
    if missing:
        print(f'[WARNING] {len(missing)} video(s) in ground truth have no predictions — treating as empty.')
        pred = list(pred) + [{'video': v, 'events': []} for v in missing]

    return pred


def main():
    parser = argparse.ArgumentParser(description='Evaluate pre-computed predictions.')
    parser.add_argument('pred_json', help='Path to prediction JSON file')
    parser.add_argument('gt_json', help='Path to ground-truth JSON file')
    parser.add_argument('--tolerances', nargs='+', type=int, default=TOLERANCES,
                        metavar='T', help='Tolerance values (default: 0 1 2 4 8)')
    parser.add_argument('--windows', nargs=2, type=int, default=WINDOWS,
                        metavar=('NMS_W', 'SNMS_W'),
                        help='NMS and SNMS window sizes (default: 1 3)')
    parser.add_argument('--nms-threshold', type=float, default=0.01,
                        help='Score threshold for NMS / SNMS (default: 0.01)')
    args = parser.parse_args()

    print(f'[INFO] Loading predictions from {args.pred_json}')
    pred = load_json(args.pred_json)

    print(f'[INFO] Loading ground truth from {args.gt_json}')
    gt = load_json(args.gt_json)

    pred = align_predictions_to_gt(pred, gt)

    split = os.path.basename(args.gt_json).split('.')[0]  # e.g. "test"
    tolerances = args.tolerances
    nms_window = args.windows[0]
    snms_window = args.windows[1]
    threshold = args.nms_threshold

    msg = ''

    # --- w/o NMS ---
    msg += f'=== Results on {split} (w/o NMS) ===\n'
    mAPs, _, tab, _ = compute_mAPs(gt, pred, tolerances=tolerances, plot_pr=False)
    avg_mAP = np.mean(mAPs)
    msg += tab + '\n'
    msg += f'Avg mAP (across tolerances): {avg_mAP * 100:.2f}\n\n'

    # --- w/ NMS ---
    msg += f'=== Results on {split} (w/ NMS{nms_window}) ===\n'
    pred_nms = non_maximum_supression(pred, window=nms_window, threshold=threshold)
    mAPs_nms, _, tab_nms, _ = compute_mAPs(gt, pred_nms, tolerances=tolerances, plot_pr=False)
    avg_mAP_nms = np.mean(mAPs_nms)
    msg += tab_nms + '\n'
    msg += f'Avg mAP (across tolerances): {avg_mAP_nms * 100:.2f}\n\n'

    # --- w/ SNMS ---
    msg += f'=== Results on {split} (w/ SNMS{snms_window}) ===\n'
    pred_snms = soft_non_maximum_supression(pred, window=snms_window, threshold=threshold)
    mAPs_snms, _, tab_snms, _ = compute_mAPs(gt, pred_snms, tolerances=tolerances, plot_pr=False)
    avg_mAP_snms = np.mean(mAPs_snms)
    msg += tab_snms + '\n'
    msg += f'Avg mAP (across tolerances): {avg_mAP_snms * 100:.2f}\n\n'

    print(msg)


if __name__ == '__main__':
    main()
