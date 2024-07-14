import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
import os
import random
from random import choices
import numpy as np
import torch

from torch.utils.data import Dataset
import argparse
import glob
import cv2
import qis_utils
from torch.utils.data import DataLoader
import memdeblur_qis_input_args


class train_dataloader(Dataset):
    def __init__(self, args):
        super(train_dataloader, self).__init__()
        self.video_paths = sorted(glob.glob(os.path.join(args.gtdata_dir, '*.mp4')))
        self.args = args
        
    def __getitem__(self, idx):
        vid_path = self.video_paths[idx]
        gt_seq = qis_utils.frames_extraction(vid_path, self.args.num_frames, start_frame = None, downsample=self.args.downsample)
        gt_seq = self.transforms(gt_seq)

        qis_seq = qis_utils.sensor_image_simulation(self.args.avg_PPP, gt_seq, self.args.QE,
                                                         self.args.theta_dark, self.args.sigma_read,
                                                         self.args.clicks_per_frame, self.args.Nbits, self.args.gain)
        
        qis_seq = qis_seq[:, None, :, :]
        gt_seq = gt_seq[:, None, :, :]
        
        qis_seq, gt_seq = qis_utils.normalize(qis_seq, max_value=(2**self.args.Nbits) - 1), qis_utils.normalize(gt_seq,
                                                                                                               max_value=255.)
        qis_seq = torch.from_numpy(qis_seq)
        gt_seq = torch.from_numpy(gt_seq)
        qis_seq = torch.cat([qis_seq, qis_seq, qis_seq], dim = 1)
        gt_seq = torch.cat([gt_seq, gt_seq, gt_seq], dim = 1)
        
        return qis_seq, gt_seq

    def __len__(self):
        return len(self.video_paths)

    def transforms(self, gt_seq):
        if self.args.transforms:
            left = random.randint(0, gt_seq.shape[2] - self.args.patch_size)
            right = left + self.args.patch_size
            top = random.randint(0, gt_seq.shape[1] - self.args.patch_size)
            bottom = top + self.args.patch_size
            gt_seq = gt_seq[:, top:bottom, left:right]

            do_nothing = lambda x: x
            flipud = lambda x: x[::-1, :]
            rot90 = lambda x: np.rot90(x, axes=(0, 1))
            rot90_flipud = lambda x: (np.rot90(x, axes=(0, 1)))[::-1, :]
            rot180 = lambda x: np.rot90(x, k=2, axes=(0, 1))
            rot180_flipud = lambda x: (np.rot90(x, k=2, axes=(0, 1)))[::-1, :]
            rot270 = lambda x: np.rot90(x, k=3, axes=(0, 1))
            rot270_flipud = lambda x: (np.rot90(x, k=3, axes=(0, 1)))[::-1, :]

            N, _, _ = gt_seq.shape
            aug_list = [do_nothing, flipud, rot90, rot90_flipud, rot180, rot180_flipud, rot270, rot270_flipud]
            w_aug = [7, 4, 4, 4, 4, 4, 4, 4]
            transf = choices(aug_list, w_aug)

            # transform all images in array
            for j in range(N):
                gt_seq[j, ...] = transf[0](gt_seq[j, ...])
        else:
            pass

        return gt_seq


class val_dataloader(Dataset):
    def __init__(self, args):
        super(val_dataloader, self).__init__()
        self.video_paths = sorted(glob.glob(os.path.join(args.valgtdata_dir, '*.mp4')))
        self.args = args

    def __getitem__(self, idx):
        vid_path = self.video_paths[idx]
        gt_seq = qis_utils.frames_extraction(vid_path, self.args.num_frames, start_frame = 50, downsample=self.args.downsample)
        
        qis_seq = qis_utils.sensor_image_simulation(self.args.avg_PPP, gt_seq, self.args.QE,
                                                         self.args.theta_dark, self.args.sigma_read,
                                                         self.args.clicks_per_frame, self.args.Nbits, self.args.gain)
        qis_seq = qis_seq[:, None, :, :]
        gt_seq = gt_seq[:, None, :, :]

        qis_seq, gt_seq = qis_utils.normalize(qis_seq, max_value=(2**self.args.Nbits) - 1), qis_utils.normalize(gt_seq,
                                                                                                               max_value=255.)
        qis_seq = torch.from_numpy(qis_seq)
        gt_seq = torch.from_numpy(gt_seq)
        qis_seq = torch.cat([qis_seq, qis_seq, qis_seq], dim = 1)
        gt_seq = torch.cat([gt_seq, gt_seq, gt_seq], dim = 1)

        return qis_seq, gt_seq

    def __len__(self):
        return len(self.video_paths)

    def transforms(self, gt_seq):
        if self.args.transforms:
            left = 30
            right = left + self.args.patch_size
            top = 30
            bottom = top + self.args.patch_size
            gt_seq = gt_seq[:, top:bottom, left:right]
        else:
            pass

        return gt_seq


def chunk_list(args, lst):
    """
    Divide a list into chunks of a given size.
    """
    return [lst[i-args.past_frames:i + args.future_frames + 1] for i in range(args.past_frames, len(lst) - args.future_frames)]


class test_dataloader(Dataset):
    def __init__(self, args, video_path):
        super(test_dataloader, self).__init__()
        self.video_path = video_path
        self.args = args

    def __getitem__(self, idx):
        gt_seq = qis_utils.frames_extraction(self.video_path, self.args.num_frames, start_frame=idx, downsample=self.args.downsample)
        qis_seq = qis_utils.sensor_image_simulation(self.args.avg_PPP, gt_seq, self.args.QE,
                                                        self.args.theta_dark, self.args.sigma_read,
                                                        self.args.clicks_per_frame, self.args.Nbits, self.args.gain)
        qis_seq = qis_seq[:, None, :, :]
        gt_seq = gt_seq[:, None, :, :]

        qis_seq, gt_seq = qis_utils.normalize(qis_seq, max_value=(2 ** self.args.Nbits) - 1), \
            qis_utils.normalize(gt_seq, max_value=255.)
        qis_seq = torch.from_numpy(qis_seq)
        gt_seq = torch.from_numpy(gt_seq)
        
        qis_seq = torch.cat([qis_seq, qis_seq, qis_seq], dim = 1)
        gt_seq = torch.cat([gt_seq, gt_seq, gt_seq], dim = 1)

        return qis_seq, gt_seq

    def __len__(self):
        vid = cv2.VideoCapture(self.video_path)
        return int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - self.args.num_frames + 1
        
class spadtest_dataloader(Dataset):
    def __init__(self, args, video_path):
        super(spadtest_dataloader, self).__init__()
        self.video_path = video_path
        self.args = args

    def __getitem__(self, idx):
        qis_seq = qis_utils.frames_extraction(self.video_path, self.args.num_frames, start_frame=idx, downsample=self.args.downsample)
        
        qis_seq = qis_seq[:, None, :, :]

        qis_seq = qis_utils.normalize(qis_seq, max_value=255.)
        qis_seq = torch.from_numpy(qis_seq)
        
        qis_seq = torch.cat([qis_seq, qis_seq, qis_seq], dim = 1)

        return qis_seq

    def __len__(self):
        vid = cv2.VideoCapture(self.video_path)
        return int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - self.args.num_frames + 1
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='qis reconstruction dataloader args')
    memdeblur_qis_input_args.memdeblur_training_args(parser)
    memdeblur_qis_input_args.sensor_args(parser)
    args = parser.parse_args()

    trainset = train_dataloader(args)
    train_dataloader = DataLoader(dataset=trainset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    qis_seq, gt_seq = next(iter(train_dataloader))
    print('')
