"""
File containing main evaluation functions
"""

#Standard imports
import copy
import glob
import json
import os
import pickle
import zipfile
from collections import defaultdict

import numpy as np
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.io import store_json

#Local imports
from util.score import compute_mAPs

#Constants
TOLERANCES = [0, 1, 2, 4, 8]
WINDOWS = [1, 3]
TOLERANCES_SN = [3, 6]
WINDOWS_SN = [3, 6]
TOLERANCES_SNB = [6, 12]
WINDOWS_SNB = [6, 12]
WINDOWS_T = [1, 3]
WINDOWS_FG = [1, 3]
INFERENCE_BATCH_SIZE = 8

class ErrorStat:

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get(self):
        return self._err / self._total

    def get_acc(self):
        return 1. - self._get()

class ForegroundF1:

    def __init__(self):
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)

    def update(self, true, pred):
        if pred != 0:
            if true != 0:
                self._tp[None] += 1
            else:
                self._fp[None] += 1

            if pred == true:
                self._tp[pred] += 1
            else:
                self._fp[pred] += 1
                if true != 0:
                    self._fn[true] += 1
        elif true != 0:
            self._fn[None] += 1
            self._fn[true] += 1

    def get(self, k):
        return self._f1(k)

    def tp_fp_fn(self, k):
        return self._tp[k], self._fp[k], self._fn[k]

    def _f1(self, k):
        denom = self._tp[k] + 0.5 * self._fp[k] + 0.5 * self._fn[k]
        if denom == 0:
            assert self._tp[k] == 0
            denom = 1
        return self._tp[k] / denom

def process_frame_predictions(dataset, classes, pred_dict, high_recall_score_threshold=0.01):

    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}
    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    err = ErrorStat()
    f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}
    h = 0
    for video, (scores, support) in (sorted(pred_dict.items())):
        label = dataset.get_labels(video)
        if np.min(support) == 0:
            support[support == 0] = 1
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        pred = np.argmax(scores, axis=1)
        err.update(label, pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        for i in range(pred.shape[0]):
            f1.update(label[i], pred[i])

            if pred[i] != 0:
                events.append({
                    'label': classes_inv[pred[i]],
                    'frame': i,
                    'score': scores[i, pred[i]].item()
                })

            for j in classes_inv:
                if scores[i, j] >= high_recall_score_threshold:
                    events_high_recall.append({
                        'label': classes_inv[j],
                        'frame': i,
                        'score': scores[i, j].item()
                    })

        pred_events.append({
            'video': video, 'events': events,
            'fps': fps_dict[video]})
        pred_events_high_recall.append({
            'video': video, 'events': events_high_recall,
            'fps': fps_dict[video]})

    return err, f1, pred_events, pred_events_high_recall, pred_scores

def non_maximum_supression(pred, window, threshold = 0.0):
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            if type(window) is not list:
                class_window = window
            else:
                class_window = window[i]
                i += 1
            while(len(v) > 0):
                e1 = max(v, key=lambda x:x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                v.pop(pos1)
                list_pos = [pos for pos, e in enumerate(v) if ((e['frame'] >= e1['frame']-class_window) & (e['frame'] <= e1['frame']+class_window))]
                for pos in list_pos[::-1]: #reverse order to avoid movement of positions in the list
                    v.pop(pos)

        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred

def soft_non_maximum_supression(pred, window, threshold = 0.01):
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            if type(window) is not list:
                class_window = window
            else:
                class_window = window[i]
                i += 1
            while(len(v) > 0):
                e1 = max(v, key=lambda x:x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                list_pos = [pos for pos, e in enumerate(v) if ((e['frame'] >= e1['frame']-class_window) & (e['frame'] <= e1['frame']+class_window))]
                for pos in list_pos:
                    v[pos]['score'] = v[pos]['score'] * (np.abs(e1['frame'] - v[pos]['frame'])) ** 2 / ((class_window+0) ** 2)
                v.pop(pos1)

        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred

def evaluate(model, dataset, split, classes, tolerances=TOLERANCES, windows=WINDOWS,
            save_pred=None, printed = True, test = False, augment=False, save_dir = None):

    pred_dict = {}
    raw_dict = {'normal': {}, 'flip': {}} # Raw prediction per clip.
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))
        raw_dict['normal'][video] = []
        raw_dict['flip'][video] = []

    # Do not up the batch size if the dataset augments
    batch_size = 1 if augment else INFERENCE_BATCH_SIZE

    # Since I don't use the horizontal flip augmentation here,
    # use batched inference for speed.

    # batch_size = INFERENCE_BATCH_SIZE
    # batch_size = 1

    print('[INFO] Inference with Batch Size: ', batch_size)

    h = 0
    feat_save = {"video": [], "start": [], "feat": []}
    chunk_cnt = 0
    dataloader = DataLoader(
            dataset, num_workers=4 * 2, pin_memory=True,
            batch_size=batch_size
    )
    for clip in tqdm(dataloader):
        if batch_size > 1:
            # Batched by dataloader
            left_patches, right_patches = clip['left_patches'], clip['right_patches']
            left_grasp, right_grasp = clip['left_grasp'], clip['right_grasp']

            _, batch_pred_scores, feat = model.predict(clip['frame'],
                                                left_patches, right_patches,
                                                left_grasp, right_grasp)

            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                _pred = feat['pred'][i] # (L, 2)
                _predD = feat['predD'][i] if feat['predD'] is not None else None # (L, )

                start = clip['start'][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    _pred = _pred[-start:, :]
                    if _predD is not None:
                        _predD = _predD[-start:]

                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:end - start, :]
                    _pred = _pred[:end - start, :]
                    if _predD is not None:
                        _predD = _predD[:end - start]

                scores[start:end, :] += pred_scores
                support[start:end] += (pred_scores.sum(axis=1) != 0) * 1
                raw_dict['normal'][video].append((_pred.cpu(), _predD.cpu() if _predD is not None else _predD, start, end))
        else:
            # Batched by dataset
            left_patches, right_patches = clip['left_patches'], clip['right_patches']
            left_grasp, right_grasp = clip['left_grasp'], clip['right_grasp']

            _, pred_scores, feat = model.predict(clip['frame'],
                                                left_patches, right_patches,
                                                left_grasp, right_grasp
                                                )

            video = clip['video'][0]
            scores, support = pred_dict[clip['video'][0]]

            _pred = feat['pred'][0] # (L, 2)
            _predD = feat['predD'][0] if feat['predD'] is not None else None # (L, )

            start = clip['start'][0].item()

            pred_scores = pred_scores[0] # to be matched with batch_size > 1 case

            if start < 0:
                pred_scores = pred_scores[-start:, :]
                _pred = _pred[-start:, :]
                if _predD is not None:
                    _predD = _predD[-start:]

                start = 0
            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:end - start, :]
                _pred = _pred[:end - start, :]
                if _predD is not None:
                    _predD = _predD[:end - start]

            scores[start:end, :] += pred_scores
            support[start:end] += (pred_scores.sum(axis=1) != 0) * 1
            raw_dict['normal'][video].append((_pred.cpu(), _predD.cpu() if _predD is not None else _predD, start, end))
            # Additional view with horizontal flip
            for i in range(1):
                start = clip['start'][0].item()
                _, pred_scores_aug, feat_aug = model.predict(clip['frame'],
                                                left_patches, right_patches,
                                                left_grasp, right_grasp,
                                                augment_inference = True)

                pred_scores_aug = pred_scores_aug[0]  # remove batch dim → (L, C)
                _pred_aug = feat_aug['pred']
                _predD_aug = feat_aug['predD'][0] if feat_aug['predD'] is not None else None

                if start < 0:
                    pred_scores_aug = pred_scores_aug[-start:, :]
                    _pred_aug = _pred_aug[-start:, :]
                    if _predD_aug is not None:
                        _predD_aug = _predD_aug[-start:]

                    start = 0
                end = start + pred_scores_aug.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores_aug = pred_scores_aug[:end - start, :]
                    _pred_aug = _pred_aug[:end - start, :]
                    if _predD_aug is not None:
                        _predD_aug = _predD_aug[:end - start]

                scores[start:end, :] += pred_scores_aug
                support[start:end] += (pred_scores_aug.sum(axis=1) != 0) * 1
                raw_dict['flip'][video].append((_pred_aug.cpu(), _predD_aug.cpu() if _predD_aug is not None else _predD_aug, start, end))

    err, f1, pred_events, pred_events_high_recall, pred_scores = \
        process_frame_predictions(dataset, classes, pred_dict, high_recall_score_threshold=0.01)

    if not test: # validation pass -> use nms?
        pred_events_high_recall = non_maximum_supression(pred_events_high_recall, window = windows[0], threshold = 0.10)
        mAPs, _, _, _ = compute_mAPs(dataset.labels, pred_events_high_recall, tolerances=tolerances, printed = True, plot_pr = False)
        avg_mAP = np.mean(mAPs)
        return avg_mAP

    else: # test pass
        msg = "=== Frame-level results ===" + '\n'
        msg += 'Error (frame-level): {:0.2f}\n'.format(err.get() * 100)

        # Frame-level confusion matrix
        def get_f1_tab_row(str_k):
            k = classes[str_k] if str_k != 'any' else None
            return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]

        rows = [get_f1_tab_row('any')]
        for c in sorted(classes):
            rows.append(get_f1_tab_row(c))

        msg += tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                        floatfmt='0.2f') + '\n\n'

        # mAP w/o NMS
        msg += '=== Results on {} (w/o NMS) ==='.format(split) + '\n'
        mAPs, _, tab, fig = compute_mAPs(dataset.labels, pred_events_high_recall,
                                         tolerances=tolerances, plot_pr = True)
        avg_mAP = np.mean(mAPs)
        msg += tab + '\n' + 'Avg mAP (across tolerances): {:0.2f}\n\n'.format(avg_mAP * 100)

        # mAP with NMS
        msg += '=== Results on {} (w/ NMS{}) ==='.format(split, str(windows[0])) + '\n'
        pred_events_high_recall_nms = non_maximum_supression(pred_events_high_recall, window = windows[0], threshold=0.01)
        mAPs, _, nms_tab, nms_fig = compute_mAPs(dataset.labels, pred_events_high_recall_nms,
                                                 tolerances=tolerances, plot_pr = True)
        avg_mAP_nms = np.mean(mAPs)
        msg += nms_tab + '\n' + 'Avg mAP (across tolerances): {:0.2f}\n\n'.format(avg_mAP_nms * 100)

        # mAP with SNMS
        msg += '=== Results on {} (w/ SNMS{}) ==='.format(split, str(windows[1])) + '\n'
        pred_events_high_recall_snms = soft_non_maximum_supression(pred_events_high_recall, window = windows[1], threshold=0.01)
        mAPs, _, snms_tab, snms_fig = compute_mAPs(dataset.labels, pred_events_high_recall_snms, tolerances=tolerances, plot_pr = True)
        avg_mAP_snms = np.mean(mAPs)
        msg += snms_tab + '\n' + 'Avg mAP (across tolerances): {:0.2f}\n\n'.format(avg_mAP_snms * 100)

        print(msg)

        if save_dir:
            # Save predictions =========
            save_pred = os.path.join(save_dir, 'pred-{}'.format(split.lower()))
            store_json(save_pred + '.json', pred_events_high_recall)
            print(f'[INFO] Storing predictions without NMS to {save_pred + ".json"}')
            store_json(save_pred + '_nms.json', pred_events_high_recall_nms)
            print(f'[INFO] Storing predictions with NMS to {save_pred + "_nms.json"}')
            store_json(save_pred + '_snms.json', pred_events_high_recall_snms)
            print(f'[INFO] Storing predictions with SNMS to {save_pred + "_snms.json"}')

            with open(save_pred + '_pred_dict.pkl', 'wb') as f:
                pickle.dump(pred_dict, f)
                print(f'[INFO] Storing sequence prediction scores to {save_pred + "_pred_dict.pkl"}')

            with open(save_pred + '_raw_dict.pkl', 'wb') as f:
                pickle.dump(raw_dict, f)
                print(f'[INFO] Storing clip prediction scores to {save_pred + "_raw_dict.pkl"}')

            # Save figures =========
            fig_dir = os.path.join(save_dir, 'figs')
            os.makedirs(fig_dir, exist_ok=True)
            # Save the result to text files:
            with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
                f.write(msg)

            fig.savefig(os.path.join(fig_dir, f'{split}_PR_Curves.png'), dpi=300, bbox_inches='tight')
            nms_fig.savefig(os.path.join(fig_dir, f'{split}_PR_Curves_NMS.png'), dpi=300, bbox_inches='tight')
            snms_fig.savefig(os.path.join(fig_dir, f'{split}_PR_Curves_SNMS.png'), dpi=300, bbox_inches='tight')
            print(f'[INFO] Storing figures to {fig_dir}')

            # # Save immediate features =========
            # with open(save_pred + '_feat.pkl', 'wb') as f:
            #     pickle.dump(feat_save, f)
            # print(f'[INFO] Storing features to {save_pred + "_feat.pkl"}')

        return mAPs, tolerances
