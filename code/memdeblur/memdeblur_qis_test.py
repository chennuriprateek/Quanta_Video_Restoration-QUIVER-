import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
import argparse
import os
import cv2
import glob
# conda_path = '/home/pchennur/.conda/envs/cent7/2020.11-py38/prateek/bin'
# os.environ['PATH'] = f'{conda_path}:{os.environ["PATH"]}'
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import memdeblur_qis_input_args
import memdeblur_qis_dataloader
import selections, qis_utils


def main(args):
    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()

    model = selections.model_select(args, args.model_name).to(args.device)
    print(f"spat Model path: {args.weights_path}")

    checkpoint = torch.load(args.weights_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
    print('weights initialized with saved model at location: %s' % args.weights_path)

    if gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(gpu_count)]).cuda()

    model.eval()

    #video_paths
    video_paths = sorted(glob.glob(os.path.join(args.testgtdata_dir, '*.mp4')))
    print(video_paths)
    for v in range(len(video_paths)):
        args.folder_name = video_paths[v].split('/')[-1]
        args.folder_name = args.folder_name.split('.')[0]
        testset = memdeblur_qis_dataloader.test_dataloader(args, video_paths[v])
        t_dataloader = DataLoader(dataset=testset, num_workers=5, batch_size=1, shuffle=False)
        psnr = test(args, t_dataloader, model)
    return 0


def test(args, dataloader, model):
    psnr = 0
    ssim = 0
    count = 0
    fidx = (args.past_frames + args.future_frames + 1) // 2
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            qis_seq, gt_seq = data
            qis_seq = (qis_seq.to(torch.float32)).to(args.device)
            gt_seq = (gt_seq.to(torch.float32)).to(args.device)
            gt_seq = gt_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            
            out_seq = model(qis_seq)
            qis_seq = qis_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            out_seq = out_seq[2]
            out_seq = out_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            
            count = count + (out_seq.shape[0] * out_seq.shape[1])
            
            psnr += qis_utils.batch_psnr(out_seq.clamp(0.0, 1.0), gt_seq.clamp(0.0, 1.0), qis_seq.clamp(0.0, 1.0), data_range=1.0, plotdir=args.plotdir, iteration=batch, visualize=args.visualize)
            
            ssim += qis_utils.batch_ssim(out_seq.clamp(0.0, 1.0), gt_seq.clamp(0.0, 1.0), data_range=1)
            
            out_seq = ((out_seq.clamp(0.0, 1.0).squeeze()) * 255).detach().cpu().numpy()
            qis_seq = ((qis_seq.clamp(0.0, 1.0).squeeze()) * 255).detach().cpu().numpy()
            gt_seq = ((gt_seq.clamp(0.0, 1.0).squeeze()) * 255).detach().cpu().numpy()
            
            out_seq = np.transpose(out_seq, (1,2,0))
            qis_seq = np.transpose(qis_seq, (1,2,0))
            gt_seq = np.transpose(gt_seq, (1,2,0))
            
            if not os.path.exists(os.path.join(args.save_path, args.folder_name + '_gt')):
                os.makedirs(os.path.join(args.save_path, args.folder_name + '_gt'))
                os.makedirs(os.path.join(args.save_path, args.folder_name + '_qis'))
                os.makedirs(os.path.join(args.save_path, args.folder_name + '_out'))

            file_name = '%05d'% (batch + fidx)

            cv2.imwrite(os.path.join(args.save_path, args.folder_name + '_qis', file_name + '_qis.png'),
                        qis_seq.astype(np.uint8))
            cv2.imwrite(os.path.join(args.save_path, args.folder_name + '_gt', file_name + '_gt.png'),
                        gt_seq.astype(np.uint8))
            cv2.imwrite(os.path.join(args.save_path, args.folder_name + '_out', file_name + '_quiverout.png'),
                        out_seq.astype(np.uint8))
            del out_seq
            del qis_seq
            del gt_seq

    print('psnr: %.2f' % (psnr/count))
    print('ssim: %.4f' % (ssim/count))
    return psnr/count  # , ssim / size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video denoising parameters')
    memdeblur_qis_input_args.memdeblur_testing_args(parser)
    memdeblur_qis_input_args.sensor_args(parser)
    args = parser.parse_args()

    model = main(args)
