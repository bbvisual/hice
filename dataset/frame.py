#!/usr/bin/env python3

"""
File containing classes related to the frame datasets.
"""

#Standard imports
import copy
import json
import math
import os
import pickle
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

from util.io import load_json, load_text
from util.score import gaussian_window

#Local imports


#Constants

# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5
FPS_SN = 25
ENLARGE_FACTOR = 1.2

"""
ActionSpotDataset -> for training/validating

ActionSpotVideoDataset -> for testing

"""
class HandAnnoHandler:
    """
    A class to handle the hand annotation logic, centralized.
    Use on both ActionSpotDataset (for train) and ActionSpotVideoDataset (for val/test)

    Input: ret with 'frame', 'hands' keys.
    What does it do?
    1. Take the hand annotations -> create probability heat map for hands appearance.
    2. Create method to handle random from cropping, horizontal flipping, and augment the hand patches
    TODO: where is the frame augmentation handling.
    TODO: fallback to center cropping when hand is missing is not ideal approach.
    """

    def __init__(self, scene_size=224, hand_size=224,
                is_training=True, flip_prob=0.5, hand_crop_prob=0.8, enlarge_factor=ENLARGE_FACTOR):
        self.scene_size = scene_size
        self.hand_size = hand_size
        self.is_training = is_training
        self.flip_prob = flip_prob
        self.hand_crop_prob = hand_crop_prob
        self.enlarge_factor = enlarge_factor
        self.center_crop = T.CenterCrop((self.scene_size, self.scene_size))

        print(f'[INFO] Enlarge factor: {self.enlarge_factor}')

    def __call__(self, ret, hand_heatmap=True):

        assert 'frame' in ret and 'hands' in ret, "[ERROR] Expected keys 'frame' and 'hands' in input."
        frames = ret['frame'] # expect (L, C, H, W)
        hands_sequence = ret['hands']
        del ret['hands']
        if 'intrinsic' in ret:
            del ret['intrinsic']
        _, _, H, W = frames.shape

        # Cropping frames, only in training
        if self.is_training:
            if random.random() > self.flip_prob:
                frames, hands_sequence = self.horizontal_flip(frames, hands_sequence)

            # Cropping frames =======
            hand_prob = self.get_hand_prob(hands_sequence, H, W)
            total = hand_prob.sum()
            # For 224p version of HOI4D: H == scene_size so H - half == half, causing
            # randint(low, low) to crash. Clamp high to at least low+1 so y is pinned
            # to center when height == crop size; x still varies freely along the wider dim.
            half = self.scene_size // 2
            y_lo, y_hi = half, max(half + 1, H - half)
            x_lo, x_hi = half, max(half + 1, W - half)
            if total <= 0 or np.isnan(total) or (hand_heatmap and random.random() < 1 - self.hand_crop_prob):
                # Random crop
                y = np.random.randint(y_lo, y_hi)
                x = np.random.randint(x_lo, x_hi)
            else:
                # hand centric cropping
                p = hand_prob / total
                p_sum = p.sum()                                  # guard against tiny drift
                if p_sum <= 0 or np.isnan(p_sum):
                    y = np.random.randint(y_lo, y_hi)
                    x = np.random.randint(x_lo, x_hi)
                else:
                    p /= p_sum
                    idx = np.random.choice(p.size, p=p)
                    y, x = np.unravel_index(idx, (H, W))

            ret['frame'] = self.crop_frames(frames, x, y)

        else:
            # Center cropping
            ret['frame'] = self.center_crop(frames)

        # Cropping hand patches =======
        ret.update(self.crop_hand_patches(frames, hands_sequence))

        return ret

    def get_hand_prob(self, hands_sequence, H, W):
        """
        From a sequence of hand annotations, create a hand heatmap from hand bounding boxes across the sequence.
        """
        hand_heatmaps = np.zeros((H, W))
        for anno in hands_sequence:
            for side in ('left hand', 'right hand'):
                box = None if anno.get(side) is None else anno[side].get('box', None)
                if box is None:
                    continue
                x1, y1, x2, y2 = map(int, box)  # Changed from x,y,w,h
                if x2 <= x1 or y2 <= y1:  # Check for valid box
                    continue
                x0 = max(0, x1)
                y0 = max(0, y1)
                x1_clamp = min(W, x2)
                y1_clamp = min(H, y2)

                if x1_clamp > x0 and y1_clamp > y0:
                    hand_heatmaps[y0:y1_clamp, x0:x1_clamp] += 1.0

        # Border to 0
        hand_heatmaps[0: self.scene_size // 2, :] = 0
        hand_heatmaps[-self.scene_size // 2:, :] = 0
        hand_heatmaps[:, 0: self.scene_size // 2] = 0
        hand_heatmaps[:, -self.scene_size // 2:] = 0

        # Make it into a probability map
        # hand_heatmaps = hand_heatmaps / (np.sum(hand_heatmaps) + 1e-6)

        # Sampling a center
        H, W = hand_heatmaps.shape
        flat = hand_heatmaps.ravel().astype(np.float64)    # ensure float
        flat[flat < 0] = 0                                  # guard: nonnegative

        return flat

    def horizontal_flip(self, frames, hands_sequence):
        """
        1. Flip the frames horizontally.
        2. Flip the hands_sequence, left to right and right to left
        """
        # (L, C, H, W) -> flip on width dimension.
        frames = torch.flip(frames, dims=[3])
        _, _, _, W = frames.shape

        # Flip the hand annotations.
        flipped_hands = []
        for anno in hands_sequence:
            new_anno = {'left hand': None, 'right hand': None}
            if anno['left hand'] is not None:
                x1, y1, x2, y2 = anno['left hand']['box']
                new_anno['right hand'] = deepcopy(anno['left hand'])
                new_anno['right hand']['box'] = [W - x2, y1, W - x1, y2]
            if anno['right hand'] is not None:
                x1, y1, x2, y2 = anno['right hand']['box']
                new_anno['left hand'] = deepcopy(anno['right hand'])
                new_anno['left hand']['box'] = [W - x2, y1, W - x1, y2]
            flipped_hands.append(new_anno)

        return frames, flipped_hands

    def get_hand_aug(self):
        """
        Hand patches are augmented with scale and translation jittering.
        There will be left hand and right hand augs.
        """
        left_scale = np.random.uniform(0.8, 1.2)
        left_tx = np.random.uniform(-0.1, 0.1)
        left_ty = np.random.uniform(-0.1, 0.1)

        right_scale = np.random.uniform(0.8, 1.2)
        right_tx = np.random.uniform(-0.1, 0.1)
        right_ty = np.random.uniform(-0.1, 0.1)

        return {'left_hand': (left_scale, left_tx, left_ty),
                'right_hand': (right_scale, right_tx, right_ty)}

    def crop_frames(self, frames, x, y, pad_value=0):
        assert frames.ndim == 4, "Expected (L, C, H, W)"
        L, C, H, W = frames.shape

        # center→top-left intended box (allow float, then floor)
        half = self.scene_size / 2.0
        x0 = int(round(x - half))
        y0 = int(round(y - half))
        x1 = x0 + self.scene_size
        y1 = y0 + self.scene_size

        # required padding on each side (left, right, top, bottom)
        pad_left   = max(0, -x0)
        pad_top    = max(0, -y0)
        pad_right  = max(0, x1 - W)
        pad_bottom = max(0, y1 - H)

        if any(v > 0 for v in (pad_left, pad_right, pad_top, pad_bottom)):
            # F.pad expects (left, right, top, bottom) for 4D (N,C,H,W)
            frames = F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

        # shift coords into the padded tensor's frame
        x0_p = x0 + pad_left
        y0_p = y0 + pad_top
        x1_p = x0_p + self.scene_size
        y1_p = y0_p + self.scene_size

        return frames[:, :, y0_p:y1_p, x0_p:x1_p]

    def crop_hand_patches(self, frames, hands_sequence):
        """
        Logic:
        1. No hand -> center crop.
        2. Has hand
            a. Jitter bbox if training. Keep it consistent across frames in the sequence (unlike the old code)
            b. Enlarge the bbox.
            c. Crop the hand patch.
            d. Get the grasp label for the hand patch.
        """
        hand_augs = self.get_hand_aug()
        left_scale, left_tx, left_ty = hand_augs['left_hand']
        right_scale, right_tx, right_ty = hand_augs['right_hand']

        res = {'left': {'patch': [], 'grasp': [], 'valid': []},
               'right': {'patch': [], 'grasp': [], 'valid': []}}
        for idx in range(len(hands_sequence)):
            for side in ['left', 'right']:
                hand_side = f'{side} hand'
                hand_data = hands_sequence[idx].get(hand_side)
                box = None if hand_data is None else hand_data.get('box', None)

                if box is not None and self.is_training:
                    x1, y1, x2, y2 = map(float, box)
                    w, h = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    # Apply jitter box
                    if side == 'left':
                        cx += left_tx * w
                        cy += left_ty * h
                        w *= left_scale
                        h *= left_scale

                    elif side == 'right':
                        cx += right_tx * w
                        cy += right_ty * h
                        w *= right_scale
                        h *= right_scale

                    box = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

                res[side]['patch'].append(self._crop_hand_patch(frames[idx], box))
                grasp_data = None if hand_data is None else hand_data.get('grasp', None)
                res[side]['grasp'].append(self._extract_grasp(grasp_data))
                res[side]['valid'].append(box is not None)

        # Restructure the dictionary and convert into torch tensor.

        return {'left_patches':     torch.stack(res['left']['patch']),
                'left_grasp':       torch.stack(res['left']['grasp']),
                'is_left_valid':    torch.tensor(res['left']['valid'], dtype=torch.int8),
                'right_patches':    torch.stack(res['right']['patch']),
                'right_grasp':      torch.stack(res['right']['grasp']),
                'is_right_valid':   torch.tensor(res['right']['valid'], dtype=torch.int8)}

    def _crop_hand_patch(self, frame, bbox):
        C, H, W = frame.shape
        if bbox is None:
            # No bbox -> central cropping with self.hand_size
            cx, cy = W / 2.0, H / 2.0
            x0, y0 = cx - self.hand_size / 2.0, cy - self.hand_size / 2.0
            x1, y1 = cx + self.hand_size / 2.0, cy + self.hand_size / 2.0
            return frame[:, int(y0):int(y1), int(x0):int(x1)].float()

        x, y, x2, y2 = map(float, bbox)
        w, h = x2 - x, y2 - y

        size = max(w, h) * self.enlarge_factor
        size = max(1.0, size)
        cx, cy = (x + x2) / 2.0, (y + y2) / 2.0

        # desired square window in original coords
        x0 = cx - size / 2.0
        y0 = cy - size / 2.0
        x1 = cx + size / 2.0
        y1 = cy + size / 2.0

        # integer sampling window
        ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
        ix1, iy1 = int(np.ceil(x1)), int(np.ceil(y1))

        # slice intersection with image
        sx0 = max(0, ix0)
        sy0 = max(0, iy0)
        sx1 = min(W, ix1)
        sy1 = min(H, iy1)
        patch = frame[:, sy0:sy1, sx0:sx1]

        # pad to the requested square if we hit borders
        pad_left   = sx0 - ix0
        pad_top    = sy0 - iy0
        pad_right  = ix1 - sx1
        pad_bottom = iy1 - sy1

        if pad_left or pad_right or pad_top or pad_bottom:
            patch = F.pad(
                patch,
                (pad_left, pad_right, pad_top, pad_bottom),  # (left,right,top,bottom)
                mode='constant', value=0.0
            )

        # resize to (self.hand_crop, self.hand_crop)
        patch = patch.float()
        patch = F.interpolate(
            patch.unsqueeze(0),
            size=(self.hand_size, self.hand_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        return patch.float()

    def _extract_grasp(self, grasp_data):
            """Extract grasp state - 8 grasp types + 1 no-grasp indicator"""
            if grasp_data is None:
                return torch.tensor([0.0] * 8 + [1.0])  # No grasp

            grasp_values = [float(v) for v in grasp_data.values()]
            return torch.tensor(grasp_values + [0.0])  # Has grasp

    def _debug(self, res, out_path='./viz_debug'):
        """
        Visualize the output of HandAnnoHandler for debugging.
        res: dict with keys 'frame', 'left_patches', 'right_patches',
            'is_left_valid', 'is_right_valid', 'left_grasp', 'right_grasp'
        """

        os.makedirs(out_path, exist_ok=True)

        frames = res['frame']               # (L, C, H, W)
        left_patches = res['left_patches']   # (L, C, hand_size, hand_size)
        right_patches = res['right_patches']
        left_valid = res['is_left_valid']    # (L,)
        right_valid = res['is_right_valid']
        left_grasp = res['left_grasp']       # (L, 9)
        right_grasp = res['right_grasp']

        L = frames.shape[0]

        for idx in range(L):
            # Convert tensors to numpy HWC uint8
            scene = frames[idx].permute(1, 2, 0).cpu().numpy()       # (H, W, 3)
            l_patch = left_patches[idx].permute(1, 2, 0).cpu().numpy()
            r_patch = right_patches[idx].permute(1, 2, 0).cpu().numpy()

            # Normalize to 0-255 if needed
            if scene.max() <= 1.0:
                scene = (scene * 255).astype(np.uint8)
                l_patch = (l_patch * 255).astype(np.uint8)
                r_patch = (r_patch * 255).astype(np.uint8)
            else:
                scene = scene.astype(np.uint8)
                l_patch = l_patch.astype(np.uint8)
                r_patch = r_patch.astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            scene = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
            l_patch = cv2.cvtColor(l_patch, cv2.COLOR_RGB2BGR)
            r_patch = cv2.cvtColor(r_patch, cv2.COLOR_RGB2BGR)

            # Resize patches to match scene height for side-by-side layout
            scene_h, scene_w = scene.shape[:2]
            patch_h = scene_h // 2

            l_patch_resized = cv2.resize(l_patch, (patch_h, patch_h))
            r_patch_resized = cv2.resize(r_patch, (patch_h, patch_h))

            # Build text panels below each patch
            text_h = 60
            panel_w = patch_h

            l_text = np.zeros((text_h, panel_w, 3), dtype=np.uint8)
            r_text = np.zeros((text_h, panel_w, 3), dtype=np.uint8)

            l_valid_str = 'Valid' if left_valid[idx] else 'Invalid'
            r_valid_str = 'Valid' if right_valid[idx] else 'Invalid'

            l_grasp_vals = left_grasp[idx].cpu().numpy()
            r_grasp_vals = right_grasp[idx].cpu().numpy()

            # Color: green if valid, red if invalid
            l_color = (0, 255, 0) if left_valid[idx] else (0, 0, 255)
            r_color = (0, 255, 0) if right_valid[idx] else (0, 0, 255)

            cv2.putText(l_text, f'L: {l_valid_str}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, l_color, 1)
            cv2.putText(l_text, f'G:{np.array2string(l_grasp_vals, precision=1, separator=",")}',
                        (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            cv2.putText(r_text, f'R: {r_valid_str}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_color, 1)
            cv2.putText(r_text, f'G:{np.array2string(r_grasp_vals, precision=1, separator=",")}',
                        (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Stack: patch on top, text below
            l_col = np.vstack([l_patch_resized, l_text])  # (patch_h + text_h, panel_w, 3)
            r_col = np.vstack([r_patch_resized, r_text])

            # Right side: left_patch and right_patch stacked vertically
            right_side = np.vstack([l_col, r_col])  # (2*(patch_h + text_h), panel_w, 3)

            # Resize scene to match right_side height
            right_h = right_side.shape[0]
            scene_resized = cv2.resize(scene, (int(scene_w * right_h / scene_h), right_h))

            # Combine
            canvas = np.hstack([scene_resized, right_side])

            # Frame label
            cv2.putText(canvas, f'Frame {idx}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imwrite(os.path.join(out_path, f'{idx:04d}.jpg'), canvas)

        print(f'[DEBUG] Saved {L} frames to {out_path}')


class ActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,                    # dict of class names to idx
            label_file,                 # path to label json
            frame_dir,                  # path to frames
            store_dir,                  # path to store files (with frames path and labels per clip)
            store_mode,                 # 'store' or 'load'
            modality,                   # [rgb, bw, flow]
            clip_len,                   # Number of frames per clip
            dataset_len,                # Number of clips
            stride=1,                   # Downsample frame rate
            overlap=1,                  # Overlap between clips (in proportion to clip_len)
            radi_displacement=0,        # Radius of displacement for labels
            soft_labels=False,          # Use soft labels (gaussian window) or not
            mixup=False,                # Mixup usage
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            crop_dim=224,
            hand_dim=224,
            dataset = 'finediving'     # Dataset name
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._split = label_file.split('/')[-1].split('.')[0]
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._dataset = dataset
        self._store_dir = store_dir

        self._store_mode = store_mode
        assert store_mode in ['store', 'load']

        self._clip_len = clip_len
        assert clip_len > 0

        self._stride = stride
        assert stride > 0

        if overlap != 1:
            self._overlap = int((1-overlap) * clip_len)
        else:
            self._overlap = 1
        assert overlap >= 0 and overlap <= 1

        self._dataset_len = dataset_len
        assert dataset_len > 0

        self._pad_len = pad_len
        assert pad_len >= 0

        self._crop_dim = crop_dim
        self._hand_dim = hand_dim

        # Label modifications
        self._radi_displacement = radi_displacement
        self._soft_labels = soft_labels
        print('[INFO] Using soft labels: ' + str(soft_labels))


        self._gaussian_labels = gaussian_window(radi_displacement, sigma=radi_displacement)

        # Mixup
        self._mixup = mixup
        assert mixup is False, '[ERROR] For hand-centric crop, mixup is not recommended to use.'

        # Frame reader class
        self._frame_reader = FrameReader(frame_dir, modality, dataset = dataset)
        self._hand_handler = HandAnnoHandler(scene_size=self._crop_dim, hand_size=self._hand_dim, is_training=True)

        #Store or load clips
        if self._store_mode == 'store':
            self._store_clips()
        elif self._store_mode == 'load':
            self._load_clips()

        self._total_len = len(self._frame_paths)

    def _store_clips(self):
        #Initialize frame paths list
        self._frame_paths = []
        self._labels_store = []
        if self._radi_displacement > 0:
            self._labelsD_store = []
        for video in tqdm(self._labels):
            video_len = int(video['num_frames'])

            labels_file = video['events']

            for base_idx in range(-self._pad_len * self._stride, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride), self._overlap):

                frames_paths = self._frame_reader.load_paths(video['video'], base_idx, base_idx + self._clip_len * self._stride, stride=self._stride)

                labels = []
                if self._radi_displacement >= 0:
                    labelsD = []
                for event in labels_file:
                    event_frame = event['frame']
                    label_idx = (event_frame - base_idx) // self._stride

                    if self._radi_displacement >= 0:
                        if (label_idx >= -self._radi_displacement and label_idx < self._clip_len + self._radi_displacement):
                            label = self._class_dict[event['label']]
                            for i in range(max(0, label_idx - self._radi_displacement), min(self._clip_len, label_idx + self._radi_displacement + 1)):
                                labels.append({'label': label, 'label_idx': i})
                                labelsD.append({'displ': i - label_idx, 'label_idx': i})

                    else: #EXCLUDE OR MODIFY FOR RADI OF 0
                        if (label_idx >= -self._dilate_len and label_idx < self._clip_len + self._dilate_len):
                            label = self._class_dict[event['label']]
                            for i in range(max(0, label_idx - self._dilate_len), min(self._clip_len, label_idx + self._dilate_len + 1)):
                                labels.append({'label': label, 'label_idx': i})

                if frames_paths[1] != -1: #in case no frames were available
                    self._frame_paths.append(frames_paths)
                    self._labels_store.append(labels)
                    if self._radi_displacement > 0:
                        self._labelsD_store.append(labelsD)

        #Save to store
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'wb') as f:
                pickle.dump(self._labelsD_store, f)
        print('[INFO] Stored clips to ' + store_path)
        return

    def _load_clips(self):
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)

        with open(store_path + '/frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + '/labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'rb') as f:
                self._labelsD_store = pickle.load(f)
        print('[INFO] Loaded clips from ' + store_path)
        return

    def _get_one(self):
        """
        Get one sample clip with labels.
        Included hand annotation.
        """
        # Get random video index
        idx = random.randint(0, self._total_len - 1)

        # Get frame_path and labels dict
        frames_path = self._frame_paths[idx]
        dict_label = self._labels_store[idx]
        if self._radi_displacement > 0:
            dict_labelD = self._labelsD_store[idx]

        # Load frames
        frames, hands = self._frame_reader.load_frames(frames_path, pad=True, stride=self._stride)

        # Process labels
        labels = np.zeros(self._clip_len, np.int64)
        if self._soft_labels:
            labels = labels.astype(np.float32)

        for label in dict_label:
            labels[label['label_idx']] = label['label']

        clip_data = {'frame': frames,
                    'contains_event': int(np.sum(labels) > 0),
                    'label': labels,
                    'hands': hands
                    }

        if self._radi_displacement > 0:
            labelsD = np.zeros(self._clip_len, np.int64)
            for label in dict_labelD:
                labelsD[label['label_idx']] = label['displ']
                if self._soft_labels:
                    labels[label['label_idx']] = self._gaussian_labels[label['displ']]
            clip_data['labelD'] = labelsD

        return clip_data

    def __getitem__(self, unused):
        ret = self._hand_handler(self._get_one())

        if self._mixup:
            # Sample another clip
            mix = self._hand_handler(self._get_one())
            ret['frame2'] = mix['frame']

            ret['contains_event2'] = mix['contains_event']
            ret['label2'] = mix['label']
            if self._radi_displacement > 0:
                ret['labelD2'] = mix['labelD']

        return ret

    def hand_centric_random_crop(self, frames, hands_sequence, hand_heatmap = True):
        """
        From a sequence of frames with hand annotations
        - Create a hand heatmap from hand bounding boxes across the sequence
        - Create a probability distribution from the heatmap. Choose a random center point from distribution.
        - Crop frames around the center point. If the crop goes outside the image, pad with 0.
        - Randomly flip horizontally the whole frame sequences and hands.
        NOTE: horizontal flipping also affects the hand side (left/right) in the annotation.

        """

        # Construct hand heatmaps
        H, W = frames[0].shape[1:] # (480, 854)
        hand_heatmaps = np.zeros((H, W))
        for anno in hands_sequence:
            for side in ('left hand', 'right hand'):
                box = None if anno.get(side) is None else anno[side].get('box', None)
                if box is None:
                    continue
                x1, y1, x2, y2 = map(int, box)  # Changed from x,y,w,h
                if x2 <= x1 or y2 <= y1:  # Check for valid box
                    continue
                x0 = max(0, x1)
                y0 = max(0, y1)
                x1_clamp = min(W, x2)
                y1_clamp = min(H, y2)

                if x1_clamp > x0 and y1_clamp > y0:
                    hand_heatmaps[y0:y1_clamp, x0:x1_clamp] += 1.0

        # Border to 0
        hand_heatmaps[0: self._crop_dim // 2, :] = 0
        hand_heatmaps[-self._crop_dim // 2:, :] = 0
        hand_heatmaps[:, 0: self._crop_dim // 2] = 0
        hand_heatmaps[:, -self._crop_dim // 2:] = 0

        # Make it into a probability map
        # hand_heatmaps = hand_heatmaps / (np.sum(hand_heatmaps) + 1e-6)

        # Sampling a center
        H, W = hand_heatmaps.shape
        flat = hand_heatmaps.ravel().astype(np.float64)    # ensure float
        flat[flat < 0] = 0                                  # guard: nonnegative
        total = flat.sum()

        if (total <= 0) or np.isnan(total) or (random.random() < 0.2) or not hand_heatmap:
            # uniform fallback inside valid region
            y = np.random.randint(self._crop_dim // 2, H - self._crop_dim // 2)
            x = np.random.randint(self._crop_dim // 2, W - self._crop_dim // 2)
        else:
            p = flat / total
            p_sum = p.sum()                                  # guard against tiny drift
            if p_sum <= 0 or np.isnan(p_sum):
                y = np.random.randint(self._crop_dim // 2, H - self._crop_dim // 2)
                x = np.random.randint(self._crop_dim // 2, W - self._crop_dim // 2)
            else:
                p /= p_sum
                idx = np.random.choice(p.size, p=p)
                y, x = np.unravel_index(idx, (H, W))

        if random.random() > 0.5:
            # Horizontal Flip
            x = W - 1 - x
            # Flip hand_sequence
            hands_sequence = self.flip_hand_sequence(hands_sequence, W)
            # Flip frames
            frames = torch.flip(frames, dims=[3])  # flip width dimension

        return (x, y), frames, hands_sequence

    def flip_hand_sequence(self, hands_sequence, W):
        """
        Horizontally flip the hand sequences and swap left/right hand annotations accordingly.
        """
        res = []
        for anno in hands_sequence:
            new_anno = {'left hand': None, 'right hand': None}
            if anno['left hand'] is not None:
                x1, y1, x2, y2 = anno['left hand']['box']
                new_anno['right hand'] = deepcopy(anno['left hand'])
                new_anno['right hand']['box'] = [W - x2, y1, W - x1, y2]
            if anno['right hand'] is not None:
                x1, y1, x2, y2 = anno['right hand']['box']
                new_anno['left hand'] = deepcopy(anno['right hand'])
                new_anno['left hand']['box'] = [W - x2, y1, W - x1, y2]
            res.append(new_anno)
        # FIXME: mask is not flipped yet.
        return res

    def center_crop_with_pad(self, frames: torch.Tensor, x: float, y: float, size: int, pad_value: float = 0.0):
        """
        frames: (L, C, H, W) torch tensor
        (x, y): crop center in pixel coords (width, height). Can be float.
        size: output crop size (square)
        pad_value: value to pad with when crop spills outside
        Returns: (L, C, size, size)

        Method: find the xyxy coords of patches -> pad the frames if patch goes outside
        Then crop the patch from the padded frames.
        """
        assert frames.ndim == 4, "Expected (L, C, H, W)"
        L, C, H, W = frames.shape

        # center→top-left intended box (allow float, then floor)
        half = size / 2.0
        x0 = int(round(x - half))
        y0 = int(round(y - half))
        x1 = x0 + size
        y1 = y0 + size

        # required padding on each side (left, right, top, bottom)
        pad_left   = max(0, -x0)
        pad_top    = max(0, -y0)
        pad_right  = max(0, x1 - W)
        pad_bottom = max(0, y1 - H)

        if any(v > 0 for v in (pad_left, pad_right, pad_top, pad_bottom)):
            # F.pad expects (left, right, top, bottom) for 4D (N,C,H,W)
            frames = F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

        # shift coords into the padded tensor's frame
        x0_p = x0 + pad_left
        y0_p = y0 + pad_top
        x1_p = x0_p + size
        y1_p = y0_p + size

        return frames[:, :, y0_p:y1_p, x0_p:x1_p]

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)

    def __len__(self):
        return self._dataset_len

class FrameReader:
    def __init__(self, frame_dir, modality, dataset):
        self._frame_dir = frame_dir
        self.modality = modality
        self.dataset = dataset

        hand_anno_dir = os.path.join(os.path.dirname(frame_dir), 'hand_anno.json')
        with open(hand_anno_dir, 'r') as f:
            self._hand_anno = json.load(f)
            print(f'[INFO] Loaded hand annotation from {hand_anno_dir} with {len(self._hand_anno)} videos.')

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path) #.float() / 255 -> into model normalization / augmentations
        return img

    def load_paths(self, video_name, start, end, stride=1, source_info = None):

        found_start = -1
        pad_start = 0
        pad_end = 0
        for frame_num in range(start, end, stride):

            if frame_num < 0:
                pad_start += 1
                continue

            if pad_end > 0:
                pad_end += 1
                continue

            frame = frame_num
            frame_path = os.path.join(self._frame_dir, video_name, f'{frame:06d}' + '.jpg')
            base_path = os.path.join(self._frame_dir, video_name)
            ndigits = -1

            exist_frame = os.path.exists(frame_path)
            if exist_frame & (found_start == -1):
                found_start = frame

            if not exist_frame:
                pad_end += 1

        ret = [base_path, found_start, pad_start, pad_end, ndigits, (end-start) // stride]

        return ret

    def load_frames(self, paths, pad=False, stride=1):
        base_path = paths[0]
        start = paths[1]
        pad_start = paths[2]
        pad_end = paths[3]
        ndigits = paths[4]
        length = paths[5]

        video = os.path.basename(base_path)

        ret = []
        hands = []

        if ndigits == -1:
            for j in range(length - pad_start - pad_end):
                frame_name = f'{start + j * stride:06d}.jpg'

                ret.append(self.read_frame(os.path.join(base_path, frame_name)))
                hands.append(self._hand_anno[video][frame_name])

        else:
            path = base_path + '/'
            for j in range(length - pad_start - pad_end):
                frame_name = str(start + j * stride).zfill(ndigits) + '.jpg'
                ret.append(self.read_frame(path + frame_name))
                hands.append(self._hand_anno[video][frame_name])

        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4)) # (B, C, H, W)

        # Always pad start, but only pad end if requested
        if pad_start > 0 or (pad and pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))

            empty_anno = {'left hand': None, 'right hand': None}
            hands = [empty_anno] * pad_start + hands + [empty_anno] * (pad_end if pad else 0)

        return ret, hands

class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            overlap_len=0,
            stride=1,
            crop_dim=224,
            hand_dim=224,
            pad_len=DEFAULT_PAD_LEN,
            dataset = 'finediving',
            hand_anno_path=None,
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride
        self._dataset = dataset

        self._frame_reader = FrameReaderVideo(frame_dir, modality, dataset=dataset, hand_anno_path=hand_anno_path)
        self._crop_dim = crop_dim
        self._hand_dim = hand_dim
        self._hand_handler = HandAnnoHandler(scene_size=self._crop_dim, hand_size=self._hand_dim, is_training=False)

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                self._clips.append((l['video'], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):

        video_name, start = self._clips[idx]
        frames, hands = self._frame_reader.load_frames(
            video_name, start, start + self._clip_len * self._stride, pad=True,
            stride=self._stride)

        ret = {'video': video_name, 'start': start // self._stride, 'frame': frames, 'hands': hands}
        self._hand_handler(ret)

        # Checking for shape consistency here.
        # if not hasattr(self, '_ref_shapes'):
        #     self._ref_shapes = {
        #         k: v.shape for k, v in ret.items() if hasattr(v, 'shape')
        #     }
        #     print(f"[DEBUG] Reference shapes (idx={idx}, video={video_name}):")
        #     for k, s in self._ref_shapes.items():
        #         print(f"  {k}: {s}")
        # else:
        #     for k, v in ret.items():
        #         if hasattr(v, 'shape') and k in self._ref_shapes:
        #             if v.shape != self._ref_shapes[k]:
        #                 print(f"⚠️ idx={idx} video={video_name} start={start}: "
        #                       f"{k} expected {self._ref_shapes[k]} got {v.shape}")

        return ret

    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        labels_file = meta['events']
        labels_half = 0

        num_frames = meta['num_frames']
        num_labels = math.ceil(num_frames / self._stride)

        labels = np.zeros(num_labels, np.int64)
        for event in labels_file:
            frame = event['frame']
            half = 0
            if (half == labels_half):
                if (frame < num_frames):
                    labels[frame // self._stride] = self._class_dict[event['label']]
                else:
                    print('Warning: {} >= {} is past the end {}'.format(
                        frame, num_frames, meta['video']))
        return labels

    @property
    def videos(self):
        return sorted([
            (v['video'], math.ceil(v['num_frames'] / self._stride),
            v.get('fps', 1) / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)

                x_copy['fps'] = x_copy.get('fps', 1) / self._stride
                x_copy['num_frames'] //= self._stride

                for e in x_copy['events']:
                    e['frame'] //= self._stride

                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])
        #num_events = sum([len(x['events']) for x in self._labels])
        #print('{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg'.format(
        #    self._src_file, len(self._labels), num_frames, self._stride,
        #    num_events / num_frames * 100))
        print('{} : {} videos, {} frames ({} stride)'.format(
            self._src_file, len(self._labels), num_frames, self._stride)
        )

class FrameReaderVideo:

    def __init__(self, frame_dir, modality, dataset, hand_anno_path=None):
        self._frame_dir = frame_dir
        self._modality = modality
        assert self._modality == 'rgb'
        self._dataset = dataset

        hand_anno_dir = hand_anno_path if hand_anno_path is not None \
            else os.path.join(os.path.dirname(frame_dir), 'hand_anno.json')
        with open(hand_anno_dir, 'r') as f:
            self._hand_anno = json.load(f)
            print(f'[INFO] Loaded hand annotation from {hand_anno_dir} with {len(self._hand_anno)} videos.')

        # Ensure hands with bbox but no grasp are included in cross-attention.
        # Grasp values are irrelevant for inference — only their presence matters.
        dummy_grasp = {str(i): 0.0 for i in range(8)}
        for video in self._hand_anno.values():
            for frame_anno in video.values():
                for side in ('left hand', 'right hand'):
                    hand = frame_anno.get(side)
                    if hand is not None and hand.get('box') is not None \
                            and hand.get('grasp') is None:
                        hand['grasp'] = dummy_grasp

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path) #/ 255 -> modified for ActionSpotVideoDataset (to be compatible with train reading without / 255)
        return img

    def load_frames(self, video_name, start, end, pad=False, stride=1, source_info = None):
        ret = []
        hands = []
        n_pad_start = 0
        n_pad_end = 0

        for frame_num in range(start, end, stride):

            if frame_num < 0:
                n_pad_start += 1
                continue

            frame_path = os.path.join(
                self._frame_dir, video_name, f'{frame_num:06d}.jpg'
            )

            try:
                img = self.read_frame(frame_path)
                ret.append(img)

                video, frame_name = frame_path.split('/')[-2:]
                hands.append(self._hand_anno[video][frame_name])

            except RuntimeError:
                # print('Missing file!', frame_path)
                n_pad_end += 1

        if len(ret) == 0:
            return -1 # Return -1 if no frames were loaded

        # In the multicrop case, the shape is (B, T, C, H, W)
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))

            empty_anno = {'left hand': None, 'right hand': None}
            hands = [empty_anno] * n_pad_start + hands + [empty_anno] * (n_pad_end if pad else 0)
        return ret, hands

class ActionSpotDatasetJoint(Dataset):

    def __init__(
            self,
            dataset1,
            dataset2
    ):
        self._dataset1 = dataset1
        self._dataset2 = dataset2


    def __getitem__(self, unused):

        if random.random() < 0.5:
            data = self._dataset1.__getitem__(unused)
            data['dataset'] = 1
            return data
        else:
            data = self._dataset2.__getitem__(unused)
            data['dataset'] = 2
            return data

    def __len__(self):
        return self._dataset1.__len__() + self._dataset2.__len__()


def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        print('{} : {} videos, {} frames'.format(
            src_file, len(labels), num_frames))
