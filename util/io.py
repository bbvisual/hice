import gzip
import json
import os
import pickle
import re

import torch
import yaml

FPS_SN = 25

def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)

def load_yaml(fpath):
    with open(fpath, 'r') as fp:
        raw = fp.read()
        return yaml.safe_load(os.path.expandvars(raw))

def load_from_save(
        args, model, optimizer, scaler, lr_scheduler
):
    with open(os.path.join(args.save_dir, 'loss.json'), 'r') as fp:
        losses = json.load(fp)

    epoch = losses[-1]['epoch']
    key = 'val_mAP' if args.criterion == 'mAP' else 'val'
    best = min(losses, key=lambda x: x[key])
    best_epoch, best_criterion = best['epoch'], best[key]

    print('[INFO] Resuming training from last epoch #{}'.format(epoch))

    checkpoint_data = torch.load(os.path.join(args.save_dir, 'checkpoint_last.pt'))

    assert checkpoint_data['epoch'] == epoch, \
        '[ERROR] Saved last checkpoints do not match epoch from loss.json'

    model.load(checkpoint_data['model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
    lr_scheduler.load_state_dict(checkpoint_data['lr_state_dict'])

    return epoch, losses, best_epoch, best_criterion

def store_json_sn(pred_path, pred, stride = 1):

    i = 0
    for game in pred:
        if i % 2 == 0:
            gameDict = dict()
            gameDict['UrlLocal'] = game['video']
            gameDict['predictions'] = []
        for event in game['events']:
            eventDict = dict()
            position = int(event['frame'] / FPS_SN * 1000 * stride)
            eventDict['gameTime'] = '{} - {}:{}'.format((i % 2) + 1, position // 60000, int((position % 60000) // 1000))
            eventDict['label'] = event['label']
            eventDict['position'] = position
            eventDict['confidence'] = event['score']
            eventDict['half'] = (i % 2) + 1
            gameDict['predictions'].append(eventDict)

        if (i % 2) == 1:
            path = os.path.join('/'.join(pred_path.split('/')[:-1]) + '/preds', '/'.join(game['video'].split('/')[:-1]))
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path + '/results_spotting.json', 'w') as fp:
                json.dump(gameDict, fp, indent=4)

        i += 1

def store_json_snb(pred_path, pred, stride = 1):
    for game in pred:
        gameDict = dict()
        gameDict['UrlLocal'] = game['video']
        gameDict['predictions'] = []
        for event in game['events']:
            eventDict = dict()
            position = int(event['frame'] / FPS_SN * 1000 * stride)
            eventDict['gameTime'] = '1 - {}:{}'.format(position // 60000, int((position % 60000) // 1000))
            eventDict['label'] = event['label']
            eventDict['position'] = position
            eventDict['confidence'] = event['score']
            eventDict['half'] = 1
            gameDict['predictions'].append(eventDict)

        path = os.path.join('/'.join(pred_path.split('/')[:-1]) + '/preds', game['video'])
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/results_spotting.json', 'w') as fp:
            json.dump(gameDict, fp, indent=4)

def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines
