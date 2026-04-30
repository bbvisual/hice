#!/usr/bin/env python3
"""
Inference script for HiCE touch event detection.

Usage:
  python inference.py --checkpoint path/to/checkpoint.pt \
                      --frame      path/to/frames/ \
                      --hand_anno  path/to/hand_anno.json \
                      --out        path/to/output/ \
                      [--label     path/to/test.json] \
                      [--config    path/to/config.yaml] \
                      [--threshold 0.5] \
                      [--gpu       0]

Outputs:
  <out>/pred_touch.json          — touch events with score >= threshold, keys: no_nms / nms / snms
  <out>/mAP_calculation/pred.json       — raw high-recall predictions (>= 0.01) for mAP
  <out>/mAP_calculation/pred_nms.json   — after NMS
  <out>/mAP_calculation/pred_snms.json  — after soft NMS
  <out>/results.txt              — mAP results (only when --label has ground truth events)

  --label is optional. If omitted, videos are auto-detected from --frame.
  mAP is computed only when --label is provided and contains ground truth events.
  Use eval_from_json.py to re-evaluate mAP_calculation/ files later.
"""

import argparse
import os
import sys
import types
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning
warnings.filterwarnings('ignore', category=UnsupportedFieldAttributeWarning)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

DEMO_CONFIG = os.path.join(os.path.dirname(__file__), 'config', 'demo', 'demo.yaml')
DEMO_CLASS  = os.path.join(os.path.dirname(__file__), 'data',   'demo', 'class.txt')
BATCH_SIZE  = 8


def get_args():
    parser = argparse.ArgumentParser(description='HiCE inference')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--frame', required=True,
                        help='Path to frames directory (one subfolder per video)')
    parser.add_argument('--hand_anno', required=True,
                        help='Path to hand_anno.json')
    parser.add_argument('--out', required=True,
                        help='Output directory')
    parser.add_argument('--label', default=None,
                        help='Path to label JSON. Auto-detected from --frame if omitted. '
                             'Provide with ground truth events to compute mAP.')
    parser.add_argument('--config', default=DEMO_CONFIG,
                        help='Path to config yaml (defaults to config/demo/demo.yaml)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Score threshold for pred_touch.json (default: 0.5)')
    parser.add_argument('--gpu', default=None,
                        help='GPU id (overrides CUDA_VISIBLE_DEVICES)')
    return parser.parse_args()


def build_args(config, device):
    args = types.SimpleNamespace()
    args.device            = device
    args.modality          = config.get('modality', 'rgb')
    args.feature_arch      = config['feature_arch']
    args.temporal_arch     = config['temporal_arch']
    args.clip_len          = config['clip_len']
    args.crop_dim          = config.get('crop_dim', 224)
    args.num_classes       = config['num_classes']
    args.n_layers          = config['n_layers']
    args.sgp_ks            = config['sgp_ks']
    args.sgp_r             = config['sgp_r']
    args.radi_displacement = config['radi_displacement']
    args.amp               = config.get('amp', True)
    args.grasp_loss        = config.get('grasp_loss', False)
    args.use_kpe           = config.get('use_kpe', False)
    args.use_glb_feat      = config.get('use_glb_feat', False)
    args.share_enc         = config.get('share_enc', False)
    args.bi_interp_post    = config.get('bi_interp_post', True)
    args.temporal_shift    = config.get('temporal_shift', True)
    args.compile           = False
    args.tolerances        = config.get('tolerance', [0, 1, 2])
    args.windows           = config.get('window', [1, 3])
    args.dataset           = config.get('dataset', 'demo')

    loss_cfg = config.get('loss', {})
    args.loss_type = loss_cfg.get('type', 'ce')
    if args.loss_type == 'ce':
        args.fg_weight   = loss_cfg.get('ce', {}).get('fg_weight', 5.0)
    elif args.loss_type == 'focal':
        args.focal_alpha = loss_cfg.get('focal', {}).get('alpha', 0.9)
        args.focal_gamma = loss_cfg.get('focal', {}).get('gamma', 2.0)
    else:
        sys.exit(f'[ERROR] Unsupported loss type: {args.loss_type}')

    return args


def auto_detect_videos(frame_dir):
    videos = []
    for name in sorted(os.listdir(frame_dir)):
        video_path = os.path.join(frame_dir, name)
        if not os.path.isdir(video_path):
            continue
        num_frames = sum(1 for f in os.listdir(video_path) if f.endswith('.jpg'))
        if num_frames == 0:
            continue
        videos.append({'video': name, 'num_frames': num_frames, 'events': []})
    if not videos:
        sys.exit(f'[ERROR] No video subdirectories with .jpg frames found in {frame_dir}')
    print(f'[INFO] Auto-detected {len(videos)} videos from {frame_dir}')
    return videos


def run_inference(model, dataset, classes):
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))

    print(f'[INFO] Inference batch size: {BATCH_SIZE}')
    dataloader = DataLoader(dataset, num_workers=8, pin_memory=True, batch_size=BATCH_SIZE)

    for clip in tqdm(dataloader, desc='Inference'):
        _, batch_scores, _ = model.predict(
            clip['frame'],
            clip['left_patches'], clip['right_patches'],
            clip['left_grasp'],   clip['right_grasp'])

        for i in range(clip['frame'].shape[0]):
            video = clip['video'][i]
            scores, support = pred_dict[video]
            pred_scores = batch_scores[i]

            start = clip['start'][i].item()
            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0
            end = min(start + pred_scores.shape[0], scores.shape[0])
            pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores
            support[start:end] += (pred_scores.sum(axis=1) != 0) * 1

    return pred_dict


def apply_threshold(pred_events, threshold):
    """Filter events to only those with score >= threshold."""
    result = []
    for v in pred_events:
        result.append({
            'video': v['video'],
            'events': [e for e in v['events'] if e['score'] >= threshold]
        })
    return result


def main():
    args_cli = get_args()
    if args_cli.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

    from dataset.frame import ActionSpotVideoDataset
    from model.model import TDEEDModel
    from util.eval import (non_maximum_supression, process_frame_predictions,
                           soft_non_maximum_supression)
    from util.io import load_json, load_text, load_yaml, store_json
    from util.score import compute_mAPs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Using device: {device}')

    config = load_yaml(args_cli.config)
    args = build_args(config, device)

    # Classes: prefer dataset-specific file, fall back to demo
    class_file = os.path.join('data', args.dataset, 'class.txt')
    if not os.path.exists(class_file):
        class_file = DEMO_CLASS
    classes = {x: i + 1 for i, x in enumerate(load_text(class_file))}
    print(f'[INFO] Classes: {classes}  (from {class_file})')

    # Label JSON
    if args_cli.label is not None:
        labels = load_json(args_cli.label)
    else:
        labels = auto_detect_videos(args_cli.frame)

    os.makedirs(args_cli.out, exist_ok=True)
    map_dir = os.path.join(args_cli.out, 'mAP_calculation')
    os.makedirs(map_dir, exist_ok=True)

    tmp_label = os.path.join(args_cli.out, '_labels_tmp.json')
    store_json(tmp_label, labels)

    # Dataset
    dataset = ActionSpotVideoDataset(
        classes,
        tmp_label,
        args_cli.frame,
        args.modality,
        args.clip_len,
        overlap_len=args.clip_len // 4 * 3,
        stride=1,
        crop_dim=args.crop_dim,
        dataset=args.dataset,
        hand_anno_path=args_cli.hand_anno,
    )

    # Model
    model = TDEEDModel(args=args)
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load(state_dict)
    print(f'[INFO] Loaded checkpoint: {args_cli.checkpoint}')

    # Run inference
    pred_dict = run_inference(model, dataset, classes)

    # Build high-recall predictions (>= 0.01) for mAP
    _, _, _, pred_raw, _ = process_frame_predictions(
        dataset, classes, pred_dict, high_recall_score_threshold=0.01)
    pred_nms  = non_maximum_supression(pred_raw,  window=args.windows[0], threshold=0.01)
    pred_snms = soft_non_maximum_supression(pred_raw, window=args.windows[1], threshold=0.01)

    # Save mAP_calculation/
    store_json(os.path.join(map_dir, 'pred.json'),      pred_raw)
    store_json(os.path.join(map_dir, 'pred_nms.json'),  pred_nms)
    store_json(os.path.join(map_dir, 'pred_snms.json'), pred_snms)
    print(f'[INFO] Saved mAP_calculation files to {map_dir}')

    # Build pred_touch.json with score >= threshold
    pred_touch = {
        'threshold': args_cli.threshold,
        'no_nms': apply_threshold(pred_raw,  args_cli.threshold),
        'nms':    apply_threshold(pred_nms,  args_cli.threshold),
        'snms':   apply_threshold(pred_snms, args_cli.threshold),
    }
    store_json(os.path.join(args_cli.out, 'pred_touch.json'), pred_touch, pretty=True)
    print(f'[INFO] Saved pred_touch.json (threshold={args_cli.threshold}) to {args_cli.out}')

    # mAP evaluation if ground truth is present
    has_gt = any(len(v.get('events', [])) > 0 for v in labels)
    if has_gt:
        print('[INFO] Ground truth events found — computing mAP...')
        msg = ''
        for name, preds in [('w/o NMS', pred_raw), ('w/ NMS', pred_nms), ('w/ SNMS', pred_snms)]:
            mAPs, _, tab, _ = compute_mAPs(
                dataset.labels, preds,
                tolerances=args.tolerances, plot_pr=False, printed=False)
            msg += f'=== Results ({name}) ===\n'
            msg += tab + '\n'
            msg += 'Avg mAP: {:0.2f}\n\n'.format(np.mean(mAPs) * 100)
        print(msg)
        with open(os.path.join(args_cli.out, 'results.txt'), 'w') as f:
            f.write(msg)
        print(f'[INFO] Saved results to {os.path.join(args_cli.out, "results.txt")}')
    else:
        print('[INFO] No ground truth events — skipping mAP. '
              'Use eval_from_json.py with mAP_calculation/ files to evaluate later.')

    os.remove(tmp_label)


if __name__ == '__main__':
    main()
