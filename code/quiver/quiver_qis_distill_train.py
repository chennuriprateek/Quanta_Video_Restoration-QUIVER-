"""Knowledge distillation training for QUIVER.

Teacher: large QUIVER (e.g. n_features=64, n_blocks=12), frozen.
Student: smaller QUIVER (e.g. n_features=32, n_blocks=6), trained.

Four feature taps via forward hooks (no modifications to existing files):
  A  forward_cell.F_R output  (hf3) — RDBCell bottleneck [B, 4F, H/4, W/4]
  B  spatial_att output             — cross-frame attention  [B, 4F, H/4, W/4]
  C  forward_cell.F_h output  (s)   — recurrent hidden state [B, 2F, H/4, W/4]
  D  alignfuse outputs (warped)     — alignment embeddings   [B*T, F, H/s, W/s]

When teacher_F != student_F, lightweight Conv1x1 adapters project student
features up to teacher channel width before computing L1 loss.

Usage example
-------------
python quiver_qis_distill_train.py \
    --gtdata_dir /path/to/train \
    --valgtdata_dir /path/to/val \
    --weights_dir ./weights_student \
    --plotdir ./plots_student \
    --n_features 32 --n_blocks 6 \
    --teacher_weights ./weights_teacher/quiver_best.pth \
    --teacher_n_features 64 --teacher_n_blocks 12 \
    --lambda_kd_hf3 1.0 --lambda_kd_att 0.5 \
    --lambda_kd_hidden 0.5 --lambda_kd_warp 0.25 \
    --lambda_task 1.0
"""

from __future__ import annotations
import sys
import os
import builtins
builtins.os = os  # quiver_qis_dataloader.py uses os before importing it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))

import argparse
import copy
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import quiver_qis_input_args
import quiver_qis_distill_args
import quiver_qis_dataloader
import quiver_qis_model
import selections
import qis_utils


# ---------------------------------------------------------------------------
# Feature hook infrastructure
# ---------------------------------------------------------------------------

class FeatureStore:
    """Accumulates tensors captured by forward hooks across a forward pass."""

    def __init__(self):
        self._store: dict[str, list[torch.Tensor]] = {}

    def clear(self):
        self._store.clear()

    def append(self, key: str, tensor: torch.Tensor):
        self._store.setdefault(key, []).append(tensor.detach() if tensor.requires_grad else tensor)

    def get(self, key: str) -> list[torch.Tensor]:
        return self._store.get(key, [])


def _make_hook(store: FeatureStore, key: str):
    """Returns a forward hook that stores the module output under `key`."""
    def hook(module, inp, output):
        if isinstance(output, (tuple, list)):
            # F_R is wrapped in nn.Sequential; output is a tensor
            store.append(key, output[0] if isinstance(output, tuple) else output)
        else:
            store.append(key, output)
    return hook


def register_hooks(model: quiver_qis_model.QUIVER, store: FeatureStore) -> list:
    """Attach forward hooks to the four distillation tap points.

    Returns a list of hook handles (call h.remove() to detach).
    """
    handles = []

    # Tap A: RDBCell bottleneck — last sub-module of F_R is RDNet
    handles.append(
        model.forward_cell.F_R.register_forward_hook(_make_hook(store, 'hf3'))
    )

    # Tap B: spatial_att output
    handles.append(
        model.spatial_att.register_forward_hook(_make_hook(store, 'att'))
    )

    # Tap C: recurrent hidden state — last Conv3x3 in F_h
    handles.append(
        model.forward_cell.F_h.register_forward_hook(_make_hook(store, 'hidden'))
    )

    # Tap D: AlignFuse outputs (three GEGLU modules, one per scale)
    for idx, mod in enumerate([model.alignfuse_1, model.alignfuse_2, model.alignfuse_3]):
        handles.append(
            mod.register_forward_hook(_make_hook(store, f'warp_{idx}'))
        )

    return handles


# ---------------------------------------------------------------------------
# Channel adapter (projects student channels → teacher channels for loss)
# ---------------------------------------------------------------------------

class ChannelAdapter(nn.Module):
    """1×1 conv adapter to align student → teacher channel counts."""

    def __init__(self, student_ch: int, teacher_ch: int):
        super().__init__()
        self.adapt = nn.Conv2d(student_ch, teacher_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapt(x)


class AdapterBank(nn.Module):
    """Holds all channel adapters for every distillation tap."""

    def __init__(self, student_F: int, teacher_F: int):
        super().__init__()
        # Tap A & B: [B, 4F, H/4, W/4]
        self.hf3 = ChannelAdapter(4 * student_F, 4 * teacher_F)
        self.att = ChannelAdapter(4 * student_F, 4 * teacher_F)
        # Tap C: [B, 2F, H/4, W/4]
        self.hidden = ChannelAdapter(2 * student_F, 2 * teacher_F)
        # Tap D (×3 scales): [B*T, F, H/s, W/s]
        self.warp_0 = ChannelAdapter(student_F, teacher_F)
        self.warp_1 = ChannelAdapter(student_F, teacher_F)
        self.warp_2 = ChannelAdapter(student_F, teacher_F)

    def adapt(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return getattr(self, key)(x)


# ---------------------------------------------------------------------------
# Feature distillation loss
# ---------------------------------------------------------------------------

def feature_distill_loss(
    s_store: FeatureStore,
    t_store: FeatureStore,
    adapters: AdapterBank | None,
    lambdas: dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    """Compute weighted L1 loss between student and teacher feature stores.

    Args:
        s_store: student FeatureStore populated during the student forward pass.
        t_store: teacher FeatureStore populated during the teacher forward pass.
        adapters: AdapterBank to project student channels to teacher width, or
                  None if teacher_F == student_F.
        lambdas: per-tap loss weights.
        device: target device.
    """
    total = torch.tensor(0.0, device=device)

    def _l1_pair(s_list, t_list, key):
        loss = torch.tensor(0.0, device=device)
        is_warp = key.startswith('warp_')
        for s_feat, t_feat in zip(s_list, t_list):
            s_feat = s_feat.to(device)
            t_feat = t_feat.to(device)
            if is_warp:
                # alignfuse output: [B, T, H, W, C] → [B*T, C, H, W]
                if s_feat.dim() == 5:
                    b, t, h, w, c = s_feat.shape
                    s_feat = s_feat.reshape(b * t, h, w, c).permute(0, 3, 1, 2).contiguous()
                if t_feat.dim() == 5:
                    b, t, h, w, c = t_feat.shape
                    t_feat = t_feat.reshape(b * t, h, w, c).permute(0, 3, 1, 2).contiguous()
            if adapters is not None:
                s_feat = adapters.adapt(key, s_feat)
            loss = loss + F.l1_loss(s_feat, t_feat.detach())
        n = max(len(s_list), 1)
        return loss / n

    tap_map = {
        'hf3':    lambdas['hf3'],
        'att':    lambdas['att'],
        'hidden': lambdas['hidden'],
        'warp_0': lambdas['warp'] * 0.5,
        'warp_1': lambdas['warp'] * 0.3,
        'warp_2': lambdas['warp'] * 0.2,
    }

    for key, weight in tap_map.items():
        if weight == 0.0:
            continue
        s_list = s_store.get(key)
        t_list = t_store.get(key)
        if not s_list or not t_list:
            continue
        total = total + weight * _l1_pair(s_list, t_list, key)

    return total


# ---------------------------------------------------------------------------
# Build teacher args (copy student args, override teacher-specific fields)
# ---------------------------------------------------------------------------

def _build_teacher_args(args):
    teacher_args = copy.copy(args)
    teacher_args.n_features = args.teacher_n_features
    teacher_args.n_blocks = args.teacher_n_blocks
    return teacher_args


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()

    # ---- Dataloaders ----
    trainset = quiver_qis_dataloader.train_dataloader(args)
    valset = quiver_qis_dataloader.val_dataloader(args)
    t_dataloader = DataLoader(dataset=trainset, num_workers=0,
                              batch_size=args.batch_size, shuffle=True)
    v_dataloader = DataLoader(dataset=valset, num_workers=0,
                              batch_size=1, shuffle=True)

    # ---- Task loss ----
    train_loss_fn, _ = selections.loss_fun_select(args)

    # ---- Teacher (frozen) ----
    teacher_args = _build_teacher_args(args)
    teacher = quiver_qis_model.QUIVER(teacher_args).to(args.device)
    if args.teacher_weights:
        ckpt = torch.load(args.teacher_weights, map_location=args.device)
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        teacher.load_state_dict(state)
        print(f'Teacher weights loaded from {args.teacher_weights}')
    else:
        print('WARNING: no teacher_weights provided — teacher is randomly initialised.')
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # ---- Student ----
    student = quiver_qis_model.QUIVER(args).to(args.device)
    print(f'Student parameters: {qis_utils.count_parameters(student):,}')
    print(f'Teacher parameters: {qis_utils.count_parameters(teacher):,}')

    if args.student_weights:
        ckpt = torch.load(args.student_weights, map_location=args.device)
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        student.load_state_dict(state)
        print(f'Student weights resumed from {args.student_weights}')

    # ---- Feature hooks ----
    s_store = FeatureStore()
    t_store = FeatureStore()
    s_handles = register_hooks(student, s_store)
    t_handles = register_hooks(teacher, t_store)

    # ---- Channel adapters (only needed if widths differ) ----
    student_F = args.n_features
    teacher_F = args.teacher_n_features
    adapters: AdapterBank | None = None
    if student_F != teacher_F:
        adapters = AdapterBank(student_F, teacher_F).to(args.device)
        print(f'AdapterBank created: student F={student_F} → teacher F={teacher_F}')

    lambdas = {
        'hf3':    args.lambda_kd_hf3,
        'att':    args.lambda_kd_att,
        'hidden': args.lambda_kd_hidden,
        'warp':   args.lambda_kd_warp,
    }

    # ---- Optimizer (student + adapters) ----
    opt_params = list(student.parameters())
    if adapters is not None:
        opt_params += list(adapters.parameters())
    optimizer = optim.Adam(opt_params, lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8,
        threshold=0.0001, threshold_mode='rel', min_lr=0, eps=1e-8,
    )

    # ---- Resume state ----
    iteration = 0
    val_psnr_best = 0.0
    if args.load_model_flag and args.student_weights:
        ckpt = torch.load(args.student_weights, map_location=args.device)
        if 'iter' in ckpt:
            iteration = ckpt['iter']
            val_psnr_best = ckpt.get('best_psnr', 0.0)
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f'Resumed at iter {iteration}, best PSNR {val_psnr_best:.3f}')

    if gpu_count > 1:
        student = nn.DataParallel(student, device_ids=list(range(gpu_count))).cuda()
        teacher = nn.DataParallel(teacher, device_ids=list(range(gpu_count))).cuda()

    start_tot = time.time()
    model_saved_name = selections.model_name_select(args, 'quiver_student')

    student.train()
    for epoch in range(args.start_epoch, args.total_epochs):
        start_ep = time.time()
        print(f'Epoch {epoch + 1}/{args.total_epochs}')
        size = len(t_dataloader.dataset)

        running_task = 0.0
        running_kd = 0.0

        for batch, data in enumerate(t_dataloader):
            for p in student.parameters():
                p.grad = None
            if adapters is not None:
                for p in adapters.parameters():
                    p.grad = None

            qis_seq, gt_seq = data
            qis_seq = qis_seq.to(torch.float32).to(args.device)
            gt_seq = gt_seq.to(torch.float32).to(args.device)

            gt1_seq = gt_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            gt2_seq = F.interpolate(
                gt1_seq, size=(args.inp_ch, args.patch_size // 2, args.patch_size // 2),
                mode='trilinear', align_corners=False)
            gt3_seq = F.interpolate(
                gt1_seq, size=(args.inp_ch, args.patch_size // 4, args.patch_size // 4),
                mode='trilinear', align_corners=False)

            # ---- Teacher forward (no grad, populates t_store) ----
            t_store.clear()
            with torch.no_grad():
                teacher(qis_seq)

            # ---- Student forward (populates s_store) ----
            s_store.clear()
            preden_seq, out1, out2, out3 = student(qis_seq)

            # ---- Task loss (mirrors original training script) ----
            out_gt_loss = (
                0.85 * train_loss_fn(out1.flatten(0, 1), gt1_seq.flatten(0, 1))
                + 0.10 * train_loss_fn(out2.flatten(0, 1), gt2_seq.flatten(0, 1))
                + 0.05 * train_loss_fn(out3.flatten(0, 1), gt3_seq.flatten(0, 1))
                + 0.20 * train_loss_fn(preden_seq.flatten(0, 1), gt_seq.flatten(0, 1))
            )

            # ---- Feature distillation loss ----
            kd_loss = feature_distill_loss(s_store, t_store, adapters, lambdas, args.device)

            loss = args.lambda_task * out_gt_loss + kd_loss

            loss.backward()
            clip_grad_norm_(student.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            running_task += out_gt_loss.item()
            running_kd += kd_loss.item()

            if batch % args.log_every == 0:
                current = (batch + 1) * args.batch_size
                lr = qis_utils.get_lr(optimizer)
                print(
                    f'[{current}/{size * args.num_frames}] '
                    f'task={out_gt_loss.item():.4f} '
                    f'kd={kd_loss.item():.4f} '
                    f'total={loss.item():.4f} '
                    f'lr={lr:.6f}'
                )

            del qis_seq, gt_seq, gt1_seq, gt2_seq, gt3_seq
            del preden_seq, out1, out2, out3

            iteration += 1

            if iteration % args.save_period == 0:
                student_state = (
                    student.module.state_dict() if gpu_count > 1 else student.state_dict()
                )
                torch.save(
                    {'iter': iteration, 'best_psnr': val_psnr_best,
                     'state_dict': student_state, 'optimizer': optimizer.state_dict()},
                    os.path.join(args.weights_dir, f'quiver_student_iter_{iteration:07d}.pth')
                )

                val_psnr = validation(args, v_dataloader, student, iteration)

                if val_psnr_best <= val_psnr:
                    val_psnr_best = val_psnr
                    torch.save(student_state, model_saved_name + '_best.pth')
                    print(f'Best student model saved → {model_saved_name}_best.pth')

                scheduler.step(val_psnr)
                torch.cuda.empty_cache()
                student.train()

        ep_time = (time.time() - start_ep) / 60
        print(
            f'Epoch {epoch + 1} done | '
            f'avg task={running_task / max(batch + 1, 1):.4f} '
            f'avg kd={running_kd / max(batch + 1, 1):.4f} | '
            f'best PSNR={val_psnr_best:.3f} | {ep_time:.1f} min'
        )

    # Remove hooks
    for h in s_handles + t_handles:
        h.remove()

    total_time = (time.time() - start_tot) / 60
    print(f'Total distillation training time: {total_time:.2f} min')
    return model_saved_name, student


def validation(args, dataloader, model, iteration):
    model.eval()
    psnr = 0.0
    size = len(dataloader.dataset)
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            qis_seq, gt_seq = data
            qis_seq = qis_seq.to(torch.float32).to(args.device)
            gt_seq = gt_seq.to(torch.float32).to(args.device)
            gt_seq = gt_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]

            _, out1, _, _ = model(qis_seq)

            qis_seq = qis_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            psnr += qis_utils.batch_psnr(
                out1.clamp(0.0, 1.0), gt_seq.clamp(0.0, 1.0),
                qis_seq.clamp(0.0, 1.0),
                data_range=1.0, plotdir=args.plotdir,
                iteration=iteration, visualize=args.visualize,
            )

    val_psnr = psnr / size
    print(f'Validation PSNR: {val_psnr:.3f}')
    del qis_seq, gt_seq, out1
    return val_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QUIVER knowledge distillation')
    quiver_qis_input_args.quiver_training_args(parser)
    quiver_qis_input_args.sensor_args(parser)
    quiver_qis_distill_args.distill_args(parser)
    args = parser.parse_args()

    main(args)
