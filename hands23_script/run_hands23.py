# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import os

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

from hodetector.data import hoMapper, register_ho_pascal_voc
from hodetector.modeling import roi_heads


GRASP_TYPES = ["NP-Palm", "NP-Fin", "Pow-Pris", "Pre-Pris",
               "Pow-Circ", "Pre-Circ", "Later", "Other"]


class _Hand:
    def __init__(self, hid, bbox, side, score, grasp_scores):
        self.id = hid
        self.bbox = bbox
        self.side = "right_hand" if side == 1 else "left_hand"
        self.score = round(float(score), 2)
        self.grasp_scores = grasp_scores

    def _fmt_scores(self, keys, vals):
        return {k: str(round(float(v), 4)) for k, v in zip(keys, vals)}

    def to_raw(self):
        return {
            "hand_side": self.side,
            "hand_bbox": [str(x) for x in self.bbox],
            "hand_pred_score": str(self.score),
            "grasp_scores": self._fmt_scores(GRASP_TYPES, self.grasp_scores),
        }


def _detect(im, predictor):
    out = predictor(im)
    inst = out["instances"]
    boxes   = inst.get("pred_boxes").tensor.cpu().numpy()
    dz      = inst.get("pred_dz").cpu().numpy()
    classes = inst.get("pred_classes").cpu().numpy()
    scores  = inst.get("scores").cpu().numpy()

    hand_side    = dz[:, 5]
    grasp_scores = torch.tensor(dz[:, 10:18])

    hands, count = [], 0
    for i, cls in enumerate(classes):
        if cls != 0:
            continue
        hands.append(_Hand(count, boxes[i], hand_side[i], scores[i], grasp_scores[i]))
        count += 1
    return hands


# ── Conversion helpers (raw detections → hand_anno format) ───────────────────

def _to_int_box(bbox):
    return [int(round(float(x))) for x in bbox]


def _convert_frame(detections):
    result = {"left hand": None, "right hand": None}
    best   = {"left hand": -1.0, "right hand": -1.0}

    for det in detections:
        side  = det["hand_side"].replace("_", " ")
        score = float(det.get("hand_pred_score", 0.0))
        if score <= best.get(side, -1.0):
            continue
        result[side] = {
            "name": [side],
            "box": _to_int_box(det["hand_bbox"]) if det.get("hand_bbox") else None,
            "grasp": det.get("grasp_scores"),
        }
        best[side] = score

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def build_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.HAND      = args.hand_thresh
    cfg.FIRSTOBJ  = args.first_obj_thresh
    cfg.SECONDOBJ = args.second_obj_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(
        args.hand_thresh, args.first_obj_thresh, args.second_obj_thresh
    )
    cfg.HAND_RELA = args.hand_rela
    cfg.OBJ_RELA  = args.obj_rela
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run Hands23 on a directory of clip folders and save hand_anno.json."
    )
    parser.add_argument("--frames", required=True,
                        help="Directory whose subdirectories are clip folders of frames.")
    parser.add_argument("--output", default="hand_anno.json",
                        help="Output path for hand_anno.json.")
    parser.add_argument("--model_weights", default="./model_weights/model_hands23.pth")
    parser.add_argument("--config_file",   default="./faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml")
    parser.add_argument("--hand_thresh",       type=float, default=0.7)
    parser.add_argument("--first_obj_thresh",  type=float, default=0.5)
    parser.add_argument("--second_obj_thresh", type=float, default=0.3)
    parser.add_argument("--hand_rela", type=float, default=0.3)
    parser.add_argument("--obj_rela",  type=float, default=0.7)
    args = parser.parse_args()

    cfg       = build_cfg(args)
    predictor = DefaultPredictor(cfg)

    clip_ids = sorted(os.listdir(args.frames_dir))
    print(f"[INFO] Found {len(clip_ids)} clips in {args.frames_dir}")

    result = {}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for clip_id in tqdm(clip_ids, desc="Clips"):
        clip_dir  = os.path.join(args.frames_dir, clip_id)
        clip_anno = {}

        for frame in sorted(os.listdir(clip_dir)):
            frame_path = os.path.join(clip_dir, frame)
            im = cv2.imread(frame_path)
            if im is None:
                continue
            detections = [h.to_raw() for h in _detect(im, predictor)]
            clip_anno[frame] = _convert_frame(detections)

        result[clip_id] = clip_anno

        # checkpoint after each clip
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

        tqdm.write(f"[INFO] Done {clip_id} ({len(clip_anno)} frames)")

    print(f"[INFO] Saved {len(result)} clips → {args.output}")


if __name__ == "__main__":
    main()
