#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""

#Standard imports
import argparse
import os
import random
import sys
import time
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

import wandb

#Constants
EVAL_SPLITS = ['test']
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--compile', action='store_true', help='Use torch.compile to optimize the model')
    parser.add_argument('--gpu', type=str, nargs='+', default=['0'])
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--store', action='store_true')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Training mode')
    mode_group.add_argument('--test', action='store_true', help='Testing mode')

    parser.add_argument('--aug', action='store_true', help='Use additional horizontal flip pass during inference')
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['store_dir']
    # args.store_mode = config['store_mode']
    args.store_mode = 'store' if args.store else 'load'
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.crop_dim = config['crop_dim']
    args.dataset = config['dataset']
    args.radi_displacement = config['radi_displacement']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.acc_grad_iter = config['acc_grad_iter']
    args.mixup = config['mixup']
    args.modality = config['modality']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.start_val_epoch = config['start_val_epoch']
    args.temporal_arch = config['temporal_arch']
    args.n_layers = config['n_layers']
    args.sgp_ks = config['sgp_ks']
    args.sgp_r = config['sgp_r']

    args.criterion = config['criterion']
    args.num_workers = config['num_workers']

    if 'loss' in config:
        args.loss_type = config['loss']['type']

        if args.loss_type == 'ce':
            args.fg_weight = config['loss']['ce']['fg_weight']

        elif args.loss_type == 'focal':
            args.focal_alpha = config['loss']['focal']['alpha']
            args.focal_gamma = config['loss']['focal']['gamma']

        else:
            sys.exit('[ERROR] Unsupported loss type!')
    else:
        # Backward compatibility if loss not in config
        args.loss_type = 'ce'
        args.fg_weight = 5.0 # default value

    if args.loss_type == 'ce':
        print(f'[INFO] Using CE loss with fg_weight {args.fg_weight}')

    elif args.loss_type == 'focal':
        print(f'[INFO] Using Focal loss with alpha {args.focal_alpha} and gamma {args.focal_gamma}')

    # Optional parameters
    args.pretrain = config.get('pretrain', None)
    args.clip_grad = config.get('clip_grad', None)
    args.grasp_loss = config.get('grasp_loss', False)
    args.use_kpe = config.get('use_kpe', False)
    args.use_glb_feat = config.get('use_glb_feat', False)
    args.share_enc = config.get('share_enc', False)
    args.soft_labels = config.get('soft_labels', False)
    args.amp = config.get('amp', True)
    args.bi_interp_post = config.get('bi_interp_post', True)
    args.temporal_shift = config.get('temporal_shift: true', True)
    args.tolerances = config.get('tolerance', [0, 1, 2])
    args.windows = config.get('window', [1, 3])
    args.eval_split = config.get('eval_split', EVAL_SPLITS)

    print('\n\n===== ABLATION SETTINGS ========================================================')
    if args.grasp_loss:
        print('[INFO] Using grasp loss')
    else:
        print('[WARN] NOT using grasp loss')

    if args.soft_labels:
        print('[INFO] Using soft labels')
    else:
        print('[WARN] NOT using soft labels')

    if args.bi_interp_post:
        print('[INFO] Using bi-linear interpolation for post-processing predictions.')
    else:
        print('[WARN] Using original interpolation for post-processing predictions.')

    if args.temporal_shift:
        print('[INFO] Using temporal shift modules in the model.')
    else:
        print('[WARN] NOT using temporal shift modules in the model.')

    print('=' * 80 + '\n\n')

    if not args.amp:
        print('[WARNING] AMP is disabled, training might be slower.')

    if config.get('sam', False):
        args.sam = config['sam']['enabled']
        args.sam_rho = config['sam']['rho']
        args.sam_adaptive = config['sam']['adaptive']
        print(f'[INFO] Using SAM optimizer with rho {args.sam_rho} and adaptive {args.sam_adaptive}')
        args.amp = False
        args.acc_grad_iter = 1
        print(f'[WARNING] When using SAM, AMP is disabled and gradient accumulation is set to 1.')
        # Reason: https://github.com/davda54/sam/issues/7
    else:
        args.sam = False

    if args.test and (args.resume or args.compile):
        warnings.warn('[WARNING] Resume and Compile flags are ignored during testing.')
        args.resume = False
        args.compile = False

    return args

def main(args):
    #Set seed
    # Lib imports
    import numpy as np
    import torch

    # from SoccerNet.Evaluation.ActionSpotting import evaluate as evaluate_SN
    from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR, LinearLR
    from torch.utils.data import DataLoader

    from dataset.datasets import get_datasets
    from dataset.frame import ActionSpotVideoDataset
    from model.model import TDEEDModel
    from util.eval import evaluate

    #Local imports
    from util.io import load_from_save, load_json, load_text, load_yaml, store_json

    def get_lr_scheduler(args, optimizer, num_steps_per_epoch, sam=False):
        cosine_epochs = args.num_epochs - args.warm_up_epochs
        print('[INFO] Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
            args.warm_up_epochs, cosine_epochs))

        if sam:
            base_optimizer = optimizer.base_optimizer
        else:
            base_optimizer = optimizer

        lr_scheduler = ChainedScheduler([
                            LinearLR(base_optimizer, start_factor=0.01, end_factor=1.0,
                                    total_iters=args.warm_up_epochs * num_steps_per_epoch),
                            CosineAnnealingLR(base_optimizer,
                                num_steps_per_epoch * cosine_epochs)])
        return args.num_epochs, lr_scheduler

    print('[INFO] Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    split_paths = os.path.split(args.model)
    if len(split_paths) > 1:
        dataset_name = split_paths[1].split('_')[0]
        config_path = os.path.join('config', dataset_name, split_paths[0], split_paths[1] + '.yaml')
    else:
        config_path = os.path.join('config', args.model.split('_')[0] + '/' + args.model + '.yaml')

    if os.path.exists(config_path):
        # Prefer loading yaml if exists
        print('[INFO] Config file: ', config_path)
        config = load_yaml(config_path)

    elif os.path.exists(config_path.replace('.yaml', '.json')):
        config_path = config_path.replace('.yaml', '.json')
        print('[INFO] Config file: ', config_path)
        config = load_json(config_path)

    else:
        sys.exit('[ERROR] Config file does not exist!, last check at: {}'.format(config_path))

    args = update_args(args, config)

    assert args.batch_size % args.acc_grad_iter == 0
    if args.crop_dim <= 0:
        args.crop_dim = None

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu' # NOTE: this can be changed in data parallel if needed.

    if args.device == 'cpu' and args.amp:
        print('[WARNING] AMP enabled on CPU, can be slow and not very useful.')

    # Save config back to json file in case of losing original file
    os.makedirs(args.save_dir, exist_ok=True)
    store_json(os.path.join(args.save_dir, 'config.json'), args.__dict__, pretty=True)

    # initialize wandb
    wandb.login()
    wandb.init(project = 'Ego-Touch-TDEED',
               name = args.model + '-' + 'seed_' + str(args.seed),
               group = args.feature_arch,
               config = args.__dict__,
               dir = args.save_dir + '/wandb_logs',
               notes = f'Run at {time.strftime("%Y-%m-%d_%H-%M-%S")}',
               mode = 'online' if args.wandb else 'disabled')

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, pretrain_classes, train_data, val_data, val_data_frames = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Stop training here and rerun.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)
    loader_batch_size = args.batch_size // args.acc_grad_iter

    # Model
    model = TDEEDModel(args=args)

    # If pretrain -> 2 prediction heads
    if args.pretrain is not None:
        n_classes = [len(classes)+1, len(pretrain_classes)+1]
        model._model.update_pred_head(n_classes)
        model._num_classes = np.array(n_classes).sum()

    sam_args = {'rho': args.sam_rho, 'adaptive': args.sam_adaptive} if args.sam else None
    optimizer, scaler = model.get_optimizer(opt_args = {'lr': args.learning_rate}, sam_args=sam_args)

    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=2, worker_init_fn=worker_init_fn)

    if not args.test:
        # Load train loader
        train_loader = DataLoader(
            train_data, shuffle=False, batch_size=loader_batch_size,
            pin_memory=True, num_workers=args.num_workers,
            prefetch_factor=2, worker_init_fn=worker_init_fn)

        # Warmup schedule
        num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch, sam=args.sam)

        losses = []
        best_criterion = 0 if args.criterion == 'map' else float('inf')
        epoch = 0

        if args.resume:
            # Resume training from last checkpoint.
            epoch, losses, best_epoch, best_criterion = load_from_save(args, model, optimizer, scaler, lr_scheduler)
            epoch += 1

        print('-' * 80)
        if args.sam:
            print('START TRAINING EPOCHS (SAM Optimizer)')
        else:
            print('START TRAINING EPOCHS')

        for epoch in range(epoch, num_epochs):
            if not args.sam:
                train_loss, train_loss_dict = model.epoch(
                    train_loader, optimizer, scaler,
                    lr_scheduler=lr_scheduler,
                    acc_grad_iter=args.acc_grad_iter,
                    # fg_weight=args.fg_weight,
                    max_norm=args.clip_grad)
            else:
                train_loss = model.epoch_sam(train_loader, optimizer,
                                             lr_scheduler=lr_scheduler,
                                             # fg_weight=args.fg_weight,
                                             max_norm=args.clip_grad)

            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]['lr']

            val_loss, val_loss_dict = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter,
                                   # fg_weight=args.fg_weight
                                   )

            better = False
            val_mAP = 0
            if args.criterion == 'loss':
                if val_loss < best_criterion:
                    best_criterion = val_loss
                    better = True
            elif args.criterion == 'map':
                if epoch >= args.start_val_epoch:
                    val_mAP = evaluate(model, val_data_frames, 'VAL', classes, tolerances=args.tolerances, windows=args.windows, printed=False, test=False)

                    if val_mAP > best_criterion:
                        best_criterion = val_mAP
                        better = True

            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f} LR: {:0.8f}'.format(
                epoch, train_loss, val_loss, current_lr))
            if (args.criterion == 'map') & (epoch >= args.start_val_epoch):
                print('Val mAP: {:0.5f}'.format(val_mAP))
                if better:
                    print('New best mAP epoch!')

            losses.append({
                'epoch': epoch,
                'train': train_loss,
                'val': val_loss,
                'val_mAP': val_mAP,
                'lr': current_lr
            })

            if args.save_dir is not None:
                store_json(os.path.join(args.save_dir, 'loss.json'), losses,
                            pretty=True)

                if better:
                    torch.save({'epoch': epoch,
                                'model_state_dict':model.state_dict()},
                               os.path.join(os.getcwd(), args.save_dir, 'checkpoint_best.pt'))

                # Save last epoch
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'lr_state_dict': lr_scheduler.state_dict()},
                           os.path.join(os.getcwd(), args.save_dir, 'checkpoint_last.pt'))

            # Log to wandb
            if (args.criterion == 'map'):
                wandb.log({'losses/train_loss': train_loss, 'losses/val_loss': val_loss, 'losses/val_mAP': val_mAP})
                wandb.log({'train/main_loss': train_loss_dict['main_loss'],
                           'train/displ_loss': train_loss_dict['displ_loss'],
                           'train/grasp_loss': train_loss_dict['grasp_loss']})
                wandb.log({'val/main_loss': val_loss_dict['main_loss'],
                           'val/displ_loss': val_loss_dict['displ_loss'],
                           'val/grasp_loss': val_loss_dict['grasp_loss']})

            else:
                wandb.log({'losses/train_loss': train_loss, 'losses/val_loss': val_loss})
            wandb.log({'losses/lr': current_lr})


    ##### TESTING / INFERENCE ################################################################
    print('-' * 80)
    print('START INFERENCE')

    best_checkpoint = torch.load(os.path.join(os.getcwd(), args.save_dir, 'checkpoint_best.pt'))
    if 'model_state_dict' in best_checkpoint:
        model.load(best_checkpoint['model_state_dict'])
        print(f'[INFO] Model loaded from {os.path.join(os.getcwd(), args.save_dir, "checkpoint_best.pt")} at epoch {best_checkpoint["epoch"]}')
    else:
        # Legacy loading
        model.load(best_checkpoint)
        print(f'[INFO] Model loaded from {os.path.join(os.getcwd(), args.save_dir, "checkpoint_best.pt")}')


    for split in args.eval_split:
        print(f'[INFO] Evaluating split: {split}')
        split_path = os.path.join(
            'data', args.dataset, '{}.json'.format(split))

        stride = STRIDE

        if os.path.exists(split_path):
            split_data = ActionSpotVideoDataset(
                classes, split_path, args.frame_dir, args.modality,
                args.clip_len, overlap_len = args.clip_len // 4 * 3,  # 3/4 overlap for video dataset, 1/2 overlap for soccernet
                stride = stride, dataset = args.dataset)

            # Augmentation is only turned off with SoccerNet or SoccerNetBall.
            # Since we don't use that dataset, set to on always.

            mAPs, tolerances = evaluate(model, split_data, split.upper(), classes,
                                        tolerances=args.tolerances,  windows=args.windows,
                                        printed = True, test = True, augment = args.aug, save_dir=args.save_dir)


            for i in range(len(mAPs)):
                wandb.log({'test/mAP@' + str(tolerances[i]): mAPs[i]})
                wandb.summary['test/mAP@' + str(tolerances[i])] = mAPs[i]

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')
    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    main(args)
