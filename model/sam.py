import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# EPOCH FUNCTION IN model.py, store here for temporary archive
# def epoch_sam(self, loader, optimizer, lr_scheduler=None, fg_weight=5, max_norm=None):

#         """ Mimic the functionalities of epoch(), but with SAM optimizer setup.
#         Only for training, and AMP is not supported for now.
#         """

#         # optimizer.zero_grad()
#         self._model.train()

#         ce_kwargs = {}
#         if fg_weight != 1:
#             ce_kwargs['weight'] = torch.FloatTensor(
#                 [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)

#         epoch_loss = 0.
#         with torch.no_grad() if optimizer is None else nullcontext():
#             for batch_idx, batch in enumerate(tqdm(loader)):
#                 frame = batch['frame'].to(self.device).float()
#                 label = batch['label']
#                 label = label.to(self.device)

#                 ### Additional KPE data
#                 left_kpe = batch['left_kpe'].to(self.device).float()
#                 right_kpe = batch['right_kpe'].to(self.device).float()

#                 left_patches = batch['left_patches'].to(self.device).float()
#                 right_patches = batch['right_patches'].to(self.device).float()

#                 left_grasp = batch['left_grasp'].to(self.device).float()
#                 right_grasp = batch['right_grasp'].to(self.device).float()

#                 ### Preparing for input, in case of using mixup or double heads ##########################################

#                 if 'labelD' in batch.keys():
#                     labelD = batch['labelD'].to(self.device).float()

#                 if 'frame2' in batch.keys():
#                     frame2 = batch['frame2'].to(self.device).float()
#                     label2 = batch['label2']
#                     label2 = label2.to(self.device)

#                     if 'labelD2' in batch.keys():
#                         labelD2 = batch['labelD2'].to(self.device).float()
#                         labelD_dist = torch.zeros((labelD.shape[0], label.shape[1])).to(self.device)

#                     l = [random.betavariate(0.2, 0.2) for _ in range(frame2.shape[0])]

#                     label_dist = torch.zeros((label.shape[0], label.shape[1], self._num_classes)).to(self.device)

#                     for i in range(frame2.shape[0]):
#                         # Merging frame and label based on mixup config.
#                         frame[i] = l[i] * frame[i] + (1 - l[i]) * frame2[i]
#                         lbl1 = label[i]
#                         lbl2 = label2[i]

#                         label_dist[i, range(label.shape[1]), lbl1] += l[i]
#                         label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]

#                         if 'labelD2' in batch.keys():
#                             labelD_dist[i] = l[i] * labelD[i] + (1 - l[i]) * labelD2[i]

#                     label = label_dist
#                     if 'labelD2' in batch.keys():
#                         labelD = labelD_dist

#                 # Depends on whether mixup is used
#                 label = label.flatten() if len(label.shape) == 2 \
#                     else label.view(-1, label.shape[-1])

#                 ### Main logic of model forward pass ##########################################
#                 # First pass of SAM
#                 self.SAM_running_stats(enable=True)
#                 preds, y = self._model(frame, y = label,
#                                         left_patches=left_patches, right_patches=right_patches,
#                                         left_kpe=left_kpe, right_kpe=right_kpe,
#                                         left_grasp=left_grasp, right_grasp=right_grasp,
#                                         inference=False)
#                 pred = preds['im_feat']

#                 if 'labelD' in batch.keys():
#                     predD = preds['displ_feat']

#                 loss = 0.
#                 predictions = pred.reshape(-1, self._num_classes) # (B*L, C+1)
#                 loss += self.main_loss(predictions, label)

#                 if self._args.grasp_loss:
#                     _, _, C = left_grasp.shape
#                     loss += 0.2 * self.grasp_loss(
#                         torch.cat([preds['left_pred'], preds['right_pred']], dim=0),
#                         torch.cat([left_grasp.view(-1, C), right_grasp.view(-1, C)], dim=0),
#                         torch.cat([preds['left_valid'], preds['right_valid']], dim=0)
#                     )

#                 if 'labelD' in batch.keys():
#                     lossD = F.mse_loss(predD, labelD, reduction = 'none')
#                     lossD = (lossD).mean()
#                     loss = loss + lossD

#                 loss.backward()
#                 if max_norm is not None:
#                     torch.nn.utils.clip_grad_norm_(
#                         [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None],
#                         max_norm)

#                 optimizer.first_step(zero_grad=True)
#                 epoch_loss += loss.detach().item()

#                 # Second forward pass
#                 del preds, y # free the memory
#                 self.SAM_running_stats(enable=False)
#                 preds, y = self._model(frame, y=label,
#                                         left_patches=left_patches, right_patches=right_patches,
#                                         left_kpe=left_kpe, right_kpe=right_kpe,
#                                         left_grasp=left_grasp, right_grasp=right_grasp,
#                                         inference=False)
#                 pred = preds['im_feat']

#                 if 'labelD' in batch.keys():
#                     predD = preds['displ_feat']

#                 loss = 0.
#                 predictions = pred.reshape(-1, self._num_classes)  # (B*L, C+1)
#                 loss += self.main_loss(predictions, label)

#                 if self._args.grasp_loss:
#                     _, _, C = left_grasp.shape
#                     loss += 0.2 * self.grasp_loss(
#                         torch.cat([preds['left_pred'], preds['right_pred']], dim=0),
#                         torch.cat([left_grasp.view(-1, C), right_grasp.view(-1, C)], dim=0),
#                         torch.cat([preds['left_valid'], preds['right_valid']], dim=0)
#                     )

#                 if 'labelD' in batch.keys():
#                     lossD = F.mse_loss(predD, labelD, reduction='none')
#                     lossD = (lossD).mean()
#                     loss = loss + lossD

#                 loss.backward()
#                 if max_norm is not None:
#                     torch.nn.utils.clip_grad_norm_(
#                         [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None],
#                         max_norm)
#                 optimizer.second_step(zero_grad=True)
#                 if lr_scheduler is not None:
#                     lr_scheduler.step()

#                 # DEBUG
#                 # break

#         return epoch_loss / len(loader)     # Avg loss
