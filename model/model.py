"""
File containing the main model.
"""

#Standard imports
import math
import random
from contextlib import nullcontext

import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from timm.models import get_pretrained_cfg
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.ops import roi_align
from tqdm import tqdm

from model.loss import BCELoss, FocalLoss, FocalLossPlain

#Local imports
from model.modules import (
    ASFormerPrediction,
    BaseRGBModel,
    EDSGPMIXERLayers,
    FC2Layers,
    FCLayers,
    SingleStageTCN,
    process_double_head,
    process_labels,
    process_prediction,
    process_prediction_orig,
    step,
)
from model.shift import make_temporal_shift


class TDEEDModel(BaseRGBModel):

    class Impl(nn.Module):
        def __init__(self, args = None):
            super().__init__()
            self._modality = args.modality
            assert self._modality == 'rgb', 'Only RGB supported for now'
            in_channels = {'rgb': 3}[self._modality]

            self._temp_arch = args.temporal_arch
            # assert self._temp_arch in ['ed_sgp_mixer'], 'Only ed_sgp_mixer supported for now'

            self._radi_displacement = args.radi_displacement

            self._feature_arch = args.feature_arch
            assert 'rny' in self._feature_arch, 'Only rny supported for now'

            self._double_head = False

            if self._feature_arch.startswith(('rny002', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                print(f'[INFO] Loaded {self._feature_arch} pre-trained model.')

                feat_dim = features.head.fc.in_features

                # Recreate the classifier head without final fc layer
                self.cls_head = timm.layers.classifier.ClassifierHead(
                                    in_features=feat_dim,
                                    num_classes=0,
                                    pool_type=features.head.global_pool.pool_type,  # same pooling
                                    drop_rate=features.head.drop.p if hasattr(features.head, 'drop') else 0.0
                                    )

                # Remove the last fc layer
                features.head = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            # Add Temporal Shift Modules
            self._require_clip_len = args.clip_len
            if self._feature_arch.endswith('_gsm') and args.temporal_shift:
                make_temporal_shift(features, args.clip_len, mode='gsm')
                print(f'[INFO] Added GSM to {self._feature_arch} model.')

            elif self._feature_arch.endswith('_gsf') and args.temporal_shift:
                make_temporal_shift(features, args.clip_len, mode='gsf')
                print(f'[INFO] Added GSF to {self._feature_arch} model.')

            else:
                print(f'[INFO] No temporal shift added to {self._feature_arch} model.')

            self._features = features

            self._feat_dim = self._d
            # ------
            # Hand patch extractor
            if not args.share_enc:
                self._hand_enc = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)

                if self._feature_arch.endswith('_gsm'):
                    make_temporal_shift(self._hand_enc, args.clip_len, mode='gsm')
                    print(f'[INFO] Added GSM to {self._feature_arch} hand encoder model.')
                elif self._feature_arch.endswith('_gsf'):
                    make_temporal_shift(self._hand_enc, args.clip_len, mode='gsf')
                    print(f'[INFO] Added GSF to {self._feature_arch} hand encoder model.')
                    print(f'[INFO] Loaded {self._feature_arch} hand encoder.')

                self._hand_enc.head = nn.Identity()

            else:
                self._hand_enc = features
                print(f'[INFO] Hand patch extractor is shared with image extractor.')


            feat_dim = self._d

            # feat_conv_dim = feat_dim
            # hand_feat_dim = feat_dim
            # self.feature_conv = nn.Sequential(
            #         nn.Conv2d(feat_conv_dim, feat_dim, kernel_size=1, bias=False),
            #         nn.BatchNorm2d(feat_dim),
            #         nn.ReLU(inplace=True),
            #     )

            self._hand_cross_attn = TDEEDModel.HandCrossAttention(d_model=feat_dim, nhead=8, dropout=0.1, normalize_before=True)
            self._ffn = TDEEDModel.FFNLayer(d_model=feat_dim, dim_feedforward=feat_dim*4, dropout=0.1, activation="relu", normalize_before=True)

            # Positional encoding
            self.temp_enc = nn.Parameter(torch.normal(mean = 0, std = 1 / args.clip_len, size = (args.clip_len, self._d)))

            if self._temp_arch == 'ed_sgp_mixer':
                self._temp_fine = EDSGPMIXERLayers(feat_dim, args.clip_len, num_layers=args.n_layers, ks = args.sgp_ks, k = args.sgp_r, concat = True)
                print(f'[INFO] Loaded {self._temp_arch} temporal architecture with {args.n_layers} blocks, ks = {args.sgp_ks}, k = {args.sgp_r} and using concatenation.')
                self._pred_fine = FCLayers(self._feat_dim, args.num_classes+1)
                radi_input_dim = self._feat_dim
                print('[INFO] Using SGPMixer')
            elif self._temp_arch == 'mstcn':
                self._temp_fine = SingleStageTCN(feat_dim, 256, feat_dim, num_layers=args.n_layers, dilate=True)
                self._pred_fine = FCLayers(self._feat_dim, args.num_classes+1)
                radi_input_dim = self._feat_dim
                print(f'[INFO] Using MS-TCN (SingleStageTCN) with {args.n_layers} dilated layers.')
            elif self._temp_arch == 'asformer':
                self._temp_fine = nn.Identity()
                self._pred_fine = ASFormerPrediction(feat_dim, args.num_classes+1, num_decoders=3)
                radi_input_dim = self._feat_dim
                if self._radi_displacement > 0:
                    print('[WARN] ASFormer: displacement head will run on pre-ASFormer features.')
                print('[INFO] Using ASFormer (last decoder output).')
            else:
                self._temp_fine = TDEEDModel.GRUPrediction(feat_dim, args.num_classes+1, hidden_dim=feat_dim, num_layers=1)
                self._pred_fine = FCLayers(2 * self._feat_dim, args.num_classes+1)
                radi_input_dim = 2 * self._feat_dim
                print('[INFO] Using GRU')

            if self._radi_displacement > 0:
                # self._pred_displ = FCLayers(self._feat_dim, 1)
                self._pred_displ = FCLayers(radi_input_dim, 1)
                print(f'[INFO] Using displacement with {self._radi_displacement} radius.')

            self._grasp_loss = args.grasp_loss

            if self._grasp_loss:
                # Grasp classifier
                self.grasp_classifier = nn.Sequential(
                    nn.Linear(feat_dim, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 9),
                )
                print('[INFO] Using grasp loss.')

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

            # Augmentation at test time, only horizontal flip
            # Use in conjunction with normal inference -> average the results
            self.augmentationI = T.Compose([
                T.RandomHorizontalFlip(p = 1.0)
            ])

            #Croping in case of using it
            self.croping = args.crop_dim

            # if self.croping is not None:
            #     self.cropT = T.RandomCrop((self.croping, self.croping))
            #     self.cropI = T.CenterCrop((self.croping, self.croping))
            # else:
            #     self.cropT = torch.nn.Identity()
            #     self.cropI = torch.nn.Identity()

            # Cropping is now handled in DataSet class.
            self.cropT = torch.nn.Identity()
            # self.cropI = T.CenterCrop((self.croping, self.croping))
            self.cropI = torch.nn.Identity()


        def forward(self, frames,
                    left_patches, right_patches,
                    left_grasp, right_grasp,
                    y = None,
                    inference = False, augment_inference = False):
            # frames: (B, L, C, H, W)
            x = self.normalize(frames) # Normalize to 0-1
            B, L, C, H, W = x.shape

            left_patches = self.normalize(left_patches)
            right_patches = self.normalize(right_patches)
            _, _, Cp, Hp, Wp = left_patches.shape

            ##### Augmenting the input image #####
            if not inference: # training mode
                ### Frames #####
                # During training, frames are already cropped in the dataset
                # no need to crop and reassign H and W.

                x = self.augment(x) # augmentation per-batch
                x = self.standarize(x) # standarization imagenet stats

                ### Hand patches #####
                left_patches = self.augment(left_patches)
                left_patches = self.standarize(left_patches)    # (B, L, 3, 224, 224)

                right_patches = self.augment(right_patches)
                right_patches = self.standarize(right_patches)  # (B, L, 3, 224, 224)

            else:
                if self.croping is not None:
                    x = self.cropI(x) # same center crop for all frames
                    H, W = self.croping, self.croping
                if augment_inference:
                    # Horizontal flip augmentation for second pass inference.
                    x = self.augmentI(x)
                    left_patches = self.augmentI(left_patches)
                    right_patches = self.augmentI(right_patches)

                    # Flip the patches as its side changed.
                    left_patches, right_patches = right_patches, left_patches

                x = self.standarize(x) # (B, L, C, H, W)
                left_patches = self.standarize(left_patches)    # (B, L, 3, 224, 224)
                right_patches = self.standarize(right_patches)  # (B, L, 3, 224, 224)

            ##### Extracting visual feature #####
            # Before global pooling: (B*L, 768, 7, 7)
            im_feat = self._features(
                x.view(-1, C, H, W) # (B*L, 3, 224, 224)
            ) # (B*L, 768, 7, 7) -> important!!!

            if inference:
                feat_save = {}
                feat_save['bb'] = self.cls_head(im_feat).reshape(B, L, self._d).detach().clone().cpu()

            # Encode hand patches.
            left_feat = self._hand_enc(
                left_patches.view(-1, Cp, Hp, Wp)
            )
            right_feat = self._hand_enc(
                right_patches.view(-1, Cp, Hp, Wp)
            )
            # Both are (B*L, 768, 7, 7)

            _, _, Nc = left_grasp.shape
            left_flag = left_grasp.view(-1, Nc)[:, -1] == 0.0
            right_flag = right_grasp.view(-1, Nc)[:, -1] == 0.0

            # Right here:
            # im_feat -> (B*L, C, 7, 7)
            # left_feat -> (B*L, C, 7, 7)
            # right_feat -> (B*L, C, 7, 7)
            # right_flag, left_flag -> (B*L,) bool

            im_feat = self._hand_cross_attn(im_feat, left_feat, right_feat, left_flag, right_flag)

            # Go through FFN.
            _, C_feat, H_feat, W_feat = im_feat.shape
            im_feat_flat = im_feat.flatten(2).permute(2, 0, 1)  # (H*W, B*L, C)
            im_feat_flat = self._ffn(im_feat_flat)              # FFN expects (seq, batch, features)
            im_feat = im_feat_flat.permute(1, 2, 0).view(B*L, C_feat, H_feat, W_feat)  # Back to (B*L, C, H, W)

            # Global pooling: (B*L, 768)
            im_feat = self.cls_head(im_feat).reshape(B, L, self._d) # (B*L, 768) -> (B, L, 768)

            if inference:
                feat_save['attn'] = im_feat.detach().clone().cpu()

            im_feat = im_feat + self.temp_enc.expand(B, -1, -1) # (B, L, 768)

            if inference:
                feat_save['pe'] = im_feat.detach().clone().cpu()


            if self._temp_arch == 'ed_sgp_mixer':
                im_feat = self._temp_fine(im_feat) # (B, L, 768)
                if inference:
                    feat_save['temporal'] = im_feat.detach().clone().cpu()
                if self._radi_displacement > 0:
                    displ_feat = self._pred_displ(im_feat).squeeze(-1) # (B, L) -> regression displacement for each frame
                    im_feat = self._pred_fine(im_feat) # (B, L, num_classes+1) -> class predictions for each frame
                    # return {'im_feat': im_feat, 'displ_feat': displ_feat}, y
                    res = {'im_feat': im_feat, 'displ_feat': displ_feat}

                else:
                    im_feat = self._pred_fine(im_feat)
                    res = {'im_feat': im_feat}

                # return im_feat, y

            elif self._temp_arch in ('mstcn', 'asformer'):
                im_feat = self._temp_fine(im_feat) # (B, L, feat_dim)
                if inference:
                    feat_save['temporal'] = im_feat.detach().clone().cpu()
                if self._radi_displacement > 0:
                    displ_feat = self._pred_displ(im_feat).squeeze(-1)
                    im_feat = self._pred_fine(im_feat)
                    res = {'im_feat': im_feat, 'displ_feat': displ_feat}
                else:
                    im_feat = self._pred_fine(im_feat)
                    res = {'im_feat': im_feat}

            else:
                im_feat = self._temp_fine(im_feat) # (6, 40, 1536)
                if self._radi_displacement > 0:
                    displ_feat = self._pred_displ(im_feat).squeeze(-1)
                    im_feat = self._pred_fine(im_feat)
                    res = {'im_feat': im_feat, 'displ_feat': displ_feat}
                else:
                    im_feat = self._pred_fine(im_feat)
                    res = {'im_feat': im_feat}

            if self._grasp_loss:
                left_preds = self.grasp_classifier(
                    self.cls_head(left_feat).view(B, L, -1)) # (B*L, feat_dim, 7, 7) -> (B*L, feat_dim) -> (B, L, feat_dim)
                right_preds = self.grasp_classifier(
                    self.cls_head(right_feat).view(B, L, -1)) # (B*L, feat_dim, 7, 7) -> (B*L, feat_dim) -> (B, L, feat_dim)

                res['left_pred'] = left_preds.reshape(B*L, -1)
                res['left_valid'] = left_flag
                res['right_pred'] = right_preds.reshape(B*L, -1)
                res['right_valid'] = right_flag

            if inference:
                return res, feat_save

            return res, y


        def normalize(self, x):
            return x / 255.

        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def augmentI(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentationI(x[i])
            return x

        def standarize(self, x):
            # for i in range(x.shape[0]):
            #     x[i] = self.standarization(x[i])
            # return x

            B, L, C, H, W = x.shape
            x = x.view(B * L, C, H, W)
            x = self.standarization(x)
            return x.view(B, L, C, H, W)

        def update_pred_head(self, num_classes = [1, 1]):
            self._pred_fine = FC2Layers(self._feat_dim, num_classes)
            self._pred_fine = self._pred_fine.cuda()
            self._double_head = True

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))
            print('  CNN features:',
                sum(p.numel() for p in self._features.parameters()))
            print('  Temporal:',
                sum(p.numel() for p in self._temp_fine.parameters()))
            print('  Head:',
                sum(p.numel() for p in self._pred_fine.parameters()))
    class HandCrossAttention(nn.Module):
        def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
            super().__init__()
            # Same components as CrossAttentionLayer
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = self._get_activation_fn(activation)
            self.normalize_before = normalize_before

            # Hand-specific components
            self.spatial_pos = TDEEDModel.PositionEmbeddingSine2D(d_model//2, normalize=True)
            self.hand_identity = nn.Embedding(2, d_model)  # 0=left, 1=right

            self._reset_parameters()

        def _reset_parameters(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def _get_activation_fn(self, activation):
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

        def with_pos_embed(self, tensor, pos):
            return tensor if pos is None else tensor + pos

        def prepare_memory(self, left_hand, right_hand, has_left, has_right):
            """Prepare memory (keys/values) and positional encodings from hand features"""
            BL, C = left_hand.shape[:2]  # B*L frames

            # Always include both hands, but mask out invalid ones
            keys, values, pos_encodings = [], [], []

            # Process left hand
            left_pos = self.spatial_pos(left_hand)  # (B*L, C, 7, 7)
            left_identity = self.hand_identity(torch.tensor(0, device=left_hand.device))
            left_pos = left_pos + left_identity.view(1, -1, 1, 1)

            keys.append(left_hand.flatten(2).permute(2, 0, 1))      # (49, B*L, C)
            values.append(left_hand.flatten(2).permute(2, 0, 1))    # (49, B*L, C)
            pos_encodings.append(left_pos.flatten(2).permute(2, 0, 1))  # (49, B*L, C)

            # Process right hand
            right_pos = self.spatial_pos(right_hand)  # (B*L, C, 7, 7)
            right_identity = self.hand_identity(torch.tensor(1, device=right_hand.device))
            right_pos = right_pos + right_identity.view(1, -1, 1, 1)

            keys.append(right_hand.flatten(2).permute(2, 0, 1))     # (49, B*L, C)
            values.append(right_hand.flatten(2).permute(2, 0, 1))   # (49, B*L, C)
            pos_encodings.append(right_pos.flatten(2).permute(2, 0, 1))  # (49, B*L, C)

            # Concatenate: left_hand (49 positions) + right_hand (49 positions) = 98 total
            memory_keys = torch.cat(keys, dim=0)        # (98, B*L, C)
            memory_values = torch.cat(values, dim=0)    # (98, B*L, C)
            memory_pos = torch.cat(pos_encodings, dim=0)  # (98, B*L, C)

            # Create per-frame padding mask based on flags
            # has_left, has_right: (B*L,) boolean tensors
            padding_mask = torch.zeros(BL, 98, dtype=torch.bool, device=left_hand.device)

            # Mask left hand positions (0:49) where has_left is False
            padding_mask[~has_left, :49] = True

            # Mask right hand positions (49:98) where has_right is False
            padding_mask[~has_right, 49:] = True

            return memory_keys, memory_values, memory_pos, padding_mask

        def forward_post(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
            """Post-norm version (same as CrossAttentionLayer)"""
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                key_padding_mask=memory_key_padding_mask
            )[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
            return tgt

        def forward_pre(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
            """Pre-norm version (same as CrossAttentionLayer)"""
            tgt2 = self.norm(tgt)
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                key_padding_mask=memory_key_padding_mask
            )[0]
            tgt = tgt + self.dropout(tgt2)
            return tgt

        def forward(self, feature_map, left_hand, right_hand, has_left, has_right):
            """
            Args:
                feature_map: (B*L, C, H, W) - main feature map (batch of sequences)
                left_hand: (B*L, C, 7, 7) - left hand features
                right_hand: (B*L, C, 7, 7) - right hand features
                has_left: (B*L,) bool tensor - whether left hand is present per frame
                has_right: (B*L,) bool tensor - whether right hand is present per frame
            Returns:
                enhanced_feature_map: (B*L, C, H, W) - attention-enhanced features
            """
            BL, C, H, W = feature_map.shape

            # Check which frames have at least one hand present
            has_any_hand = has_left | has_right

            # If no hands are present in any frame, return original feature map
            if not has_any_hand.any():
                return feature_map

            # Initialize output with original features
            enhanced_output = feature_map.clone()

            # Get indices of frames with at least one hand
            valid_indices = torch.where(has_any_hand)[0]

            if len(valid_indices) == 0:
                return enhanced_output

            # Extract only valid frames for attention computation
            valid_feature_map = feature_map[valid_indices]
            valid_left_hand = left_hand[valid_indices]
            valid_right_hand = right_hand[valid_indices]
            valid_has_left = has_left[valid_indices]
            valid_has_right = has_right[valid_indices]

            # Prepare queries for valid frames only
            feat_pos = self.spatial_pos(valid_feature_map)  # (N_valid, C, H, W)
            tgt = valid_feature_map.flatten(2).permute(2, 0, 1)        # (H*W, N_valid, C)
            query_pos = feat_pos.flatten(2).permute(2, 0, 1)          # (H*W, N_valid, C)

            # Prepare memory for valid frames only
            memory_keys, memory_values, memory_pos, key_padding_mask = self.prepare_memory(
                valid_left_hand, valid_right_hand, valid_has_left, valid_has_right
            )

            # Apply cross-attention only on valid frames
            if self.normalize_before:
                enhanced = self.forward_pre(
                    tgt=tgt,
                    memory=memory_values,
                    memory_key_padding_mask=key_padding_mask,
                    pos=memory_pos,
                    query_pos=query_pos
                )
            else:
                enhanced = self.forward_post(
                    tgt=tgt,
                    memory=memory_values,
                    memory_key_padding_mask=key_padding_mask,
                    pos=memory_pos,
                    query_pos=query_pos
                )

            # Reshape enhanced valid frames back to feature map format
            enhanced_valid = enhanced.permute(1, 2, 0).view(len(valid_indices), C, H, W)

            # Place enhanced features back into output tensor at valid positions
            enhanced_output[valid_indices] = enhanced_valid

            return enhanced_output
    class PositionEmbeddingSine2D(nn.Module):
        """
        2D version of positional embedding using sine and cosine functions.
        Adapted for feature maps with shape (batch, channels, height, width).
        """

        def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
            super().__init__()
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            if scale is not None and normalize is False:
                raise ValueError("normalize should be True if scale is passed")
            if scale is None:
                scale = 2 * math.pi
            self.scale = scale

        def forward(self, x, mask=None):
            # x shape: b, c, h, w
            assert x.dim() == 4, f"{x.shape} should be a 4-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"

            if mask is None:
                mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)

            not_mask = ~mask  # shape: b, h, w
            y_embed = not_mask.cumsum(1, dtype=torch.float32)  # cumsum along height
            x_embed = not_mask.cumsum(2, dtype=torch.float32)  # cumsum along width

            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

            pos_x = x_embed[:, :, :, None] / dim_t  # b, h, w, num_pos_feats
            pos_y = y_embed[:, :, :, None] / dim_t  # b, h, w, num_pos_feats

            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # b, c, h, w
            return pos
    class FFNLayer(nn.Module):
        def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                    activation="relu", normalize_before=False):
            super().__init__()
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            self.norm = nn.LayerNorm(d_model)

            self.activation = self._get_activation_fn(activation)
            self.normalize_before = normalize_before

            self._reset_parameters()

        def _reset_parameters(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def with_pos_embed(self, tensor, pos):
            return tensor if pos is None else tensor + pos

        def forward_post(self, tgt):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
            return tgt

        def forward_pre(self, tgt):
            tgt2 = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout(tgt2)
            return tgt

        def forward(self, tgt):
            if self.normalize_before:
                return self.forward_pre(tgt)
            return self.forward_post(tgt)

        def _get_activation_fn(self, activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    class GRUPrediction(nn.Module):

        def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
            super().__init__()
            self._gru = nn.GRU(
                feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                bidirectional=True)
            # self._fc_out = TDEEDModel.FCPrediction(2 * hidden_dim, num_classes)
            # self._dropout = nn.Dropout()

        def forward(self, x):
            y, _ = self._gru(x)
            return y
            # return self._fc_out(self._dropout(y))

    class FCPrediction(nn.Module):

        def __init__(self, feat_dim, num_classes):
            super().__init__()
            self._fc_out = nn.Linear(feat_dim, num_classes)

        def forward(self, x):
            batch_size, clip_len, _ = x.shape
            return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
                batch_size, clip_len, -1)

    def __init__(self, args=None):
        self.device = args.device
        self.amp = args.amp

        self._model = TDEEDModel.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes + 1

        if args.loss_type == 'ce':
            self.main_loss = BCELoss(fg_weight=args.fg_weight)

        elif args.loss_type == 'focal':
            self.main_loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

        if args.bi_interp_post:
            self.process_prediction = process_prediction

        else:
            self.process_prediction = process_prediction_orig

        if args.compile:
            print('[INFO] Using torch.compile for model optimization.')
            self._model = torch.compile(self._model)


    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
            acc_grad_iter=1, fg_weight=5, valMAP=False, max_norm=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        if valMAP:
            map_labels = []
            map_preds = []

        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)

        epoch_loss = 0.
        loss_dict = {'main_loss': 0., 'displ_loss': 0., 'grasp_loss': 0.}
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device)

                left_patches = batch['left_patches'].to(self.device).float()
                right_patches = batch['right_patches'].to(self.device).float()

                left_grasp = batch['left_grasp'].to(self.device).float()
                right_grasp = batch['right_grasp'].to(self.device).float()

                ### Preparing for input, in case of using mixup or double heads ##########################################

                # update labels for double head
                if self._model._double_head:
                    batch_dataset = batch['dataset']
                    label = update_labels_2heads(label, batch_dataset, self._args.num_classes)

                if 'labelD' in batch.keys():
                    labelD = batch['labelD'].to(self.device).float()

                if 'frame2' in batch.keys():
                    frame2 = batch['frame2'].to(self.device).float()
                    label2 = batch['label2']
                    label2 = label2.to(self.device)

                    if 'labelD2' in batch.keys():
                        labelD2 = batch['labelD2'].to(self.device).float()
                        labelD_dist = torch.zeros((labelD.shape[0], label.shape[1])).to(self.device)

                    l = [random.betavariate(0.2, 0.2) for _ in range(frame2.shape[0])]

                    label_dist = torch.zeros((label.shape[0], label.shape[1], self._num_classes)).to(self.device)

                    for i in range(frame2.shape[0]):
                        # Merging frame and label based on mixup config.
                        frame[i] = l[i] * frame[i] + (1 - l[i]) * frame2[i]
                        lbl1 = label[i]
                        lbl2 = label2[i]

                        label_dist[i, range(label.shape[1]), lbl1] += l[i]
                        label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]

                        if 'labelD2' in batch.keys():
                            labelD_dist[i] = l[i] * labelD[i] + (1 - l[i]) * labelD2[i]

                    label = label_dist
                    if 'labelD2' in batch.keys():
                        labelD = labelD_dist

                if valMAP:
                    labels_aux = process_labels(label, labelD if 'labelD' in batch.keys() else None,
                                        num_classes = self._num_classes)
                    map_labels.append(labels_aux.cpu())

                # Depends on whether mixup is used
                label = label.flatten() if len(label.shape) == 2 \
                    else label.view(-1, label.shape[-1])

                ### Main logic of model forward pass ##########################################
                with torch.amp.autocast(device_type = self.device, enabled = self.amp):
                    # pred, y = self._model(frame, y = label, inference=inference)
                    preds, y = self._model(frame, y = label,
                                            left_patches=left_patches, right_patches=right_patches,
                                            left_grasp=left_grasp, right_grasp=right_grasp,
                                            inference=inference)
                    pred = preds['im_feat']

                    if 'labelD' in batch.keys():
                        predD = preds['displ_feat']

                    if valMAP:
                        pred_aux = self.process_prediction(pred, predD)
                        map_preds.append(pred_aux.cpu())

                    loss = 0.

                    if self._model._double_head:
                        b, t, c = pred.shape
                        if len(label.shape) == 2:
                            label = label.view(b, t, c)
                        if len(label.shape) == 1:
                            label = label.view(b, t)

                        for i in range(pred.shape[0]):
                            if batch_dataset[i] == 1:
                                if len(label.shape) == 3:
                                    aux_label = label[i][:, :self._args.num_classes+1]
                                elif len(label.shape) == 2:
                                    aux_label = label[i]
                                else:
                                    raise NotImplementedError

                                loss += F.cross_entropy(pred[i][:, :self._args.num_classes+1], aux_label,
                                                        weight = ce_kwargs['weight'][:self._args.num_classes+1]) / (pred.shape[0])

                            elif batch_dataset[i] == 2:
                                if len(label.shape) == 3:
                                    aux_label = label[i][:, self._args.num_classes+1:]
                                elif len(label.shape) == 2:
                                    aux_label = label[i] - (self._args.num_classes + 1)
                                else:
                                    raise NotImplementedError

                                loss += F.cross_entropy(pred[i][:, self._args.num_classes+1:], aux_label,
                                                        weight = ce_kwargs['weight'][:self._args.pretrain['num_classes']+1]) / (pred.shape[0])

                    else:
                        predictions = pred.reshape(-1, self._num_classes) # (B*L, C+1)

                        # loss += F.cross_entropy(predictions, label, **ce_kwargs)
                        main_loss = self.main_loss(predictions, label)
                        loss += main_loss
                        loss_dict['main_loss'] += main_loss.detach().item()

                        if self._args.grasp_loss:
                            _, _, C = left_grasp.shape
                            grasp_loss = 0.2 * self.grasp_loss(
                                torch.cat([preds['left_pred'], preds['right_pred']], dim=0),
                                torch.cat([left_grasp.view(-1, C), right_grasp.view(-1, C)], dim=0),
                                torch.cat([preds['left_valid'], preds['right_valid']], dim=0)
                            )
                            loss += grasp_loss
                            loss_dict['grasp_loss'] += grasp_loss.detach().item()

                    if 'labelD' in batch.keys():
                        lossD = F.mse_loss(predD, labelD, reduction = 'none')
                        lossD = (lossD).mean()
                        loss += lossD
                        loss_dict['displ_loss'] += lossD.detach().item()

                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter,
                        lr_scheduler=lr_scheduler,
                        backward_only=(batch_idx + 1) % acc_grad_iter != 0,
                        max_norm=max_norm)

                epoch_loss += loss.detach().item()

                # break # DEBUG

        if valMAP:
            return epoch_loss / len(loader), torch.cat(map_labels, 0), torch.cat(map_preds, 0)

        return epoch_loss / len(loader), {k: v / len(loader) for k, v in loss_dict.items()}     # Avg loss

    def grasp_loss(self, pred_grasp, gt_grasp, is_valid, return_mean=True):
        loss_grasp = F.cross_entropy(pred_grasp, gt_grasp, reduction='none')
        loss_grasp = loss_grasp * is_valid
        if return_mean:
            loss_grasp = loss_grasp.mean()
        return loss_grasp

    def predict(self, seq,
                left_patches = None, right_patches = None,
                left_grasp = None, right_grasp = None,
                use_amp=True, augment_inference = False):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
            left_patches = left_patches.to(self.device)
            right_patches = right_patches.to(self.device)
            left_grasp = left_grasp.to(self.device)
            right_grasp = right_grasp.to(self.device)

        seq = seq.float()

        raw_pred = {'pred': None, 'predD': None}

        self._model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda') if use_amp else nullcontext():
                _pred, y = self._model(seq, y=None,
                                      left_patches=left_patches, right_patches=right_patches,
                                      left_grasp=left_grasp, right_grasp=right_grasp,
                                      inference=True, augment_inference=augment_inference)
            if isinstance(_pred, dict):
                pred = _pred['im_feat']
                if isinstance(pred, list):
                    pred = pred[0]

                if 'displ_feat' in _pred:
                    predD = _pred['displ_feat']
                    if isinstance(predD, list):
                        predD = predD[0]

                    raw_pred['predD'] = predD
                    if self._model._double_head:
                        pred = process_double_head(pred, predD, num_classes = self._args.num_classes+1)
                    else:
                        pred = self.process_prediction(pred, predD)

                raw_pred['pred'] = pred
                pred_cls = torch.argmax(pred, axis=2)
                return pred_cls.cpu().numpy(), pred.cpu().numpy(), raw_pred

            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            else:
                pred = torch.softmax(pred, axis=2)

            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy(), y

    def SAM_running_stats(self, enable=True):
        """
        From davda54's SAM implementation.
        https://github.com/davda54/sam/blob/main/example/utility/bypass_bn.py
        """
        def _enable(module):
            if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        if enable:
            self._model.apply(_enable)
        else:
            self._model.apply(_disable)

def update_labels_2heads(labels, datasets, num_classes1 = 1):
    for i in range(len(datasets)):
        if datasets[i] == 2:
            labels[i] = labels[i] + num_classes1 + 1

    return labels
