import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
import argparse
import time
import os
# conda_path = '/home/pchennur/.conda/envs/cent7/2020.02-py37/rtm1/bin'
# os.environ['PATH'] = f'{conda_path}:{os.environ["PATH"]}'
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from fvcore.nn import FlopCountAnalysis
import quiver_qis_input_args
import quiver_qis_dataloader
import selections, qis_utils


def main(args):
    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()

    start_tot = time.time()
    trainset = quiver_qis_dataloader.train_dataloader(args)
    valset = quiver_qis_dataloader.val_dataloader(args)

    t_dataloader = DataLoader(dataset=trainset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    v_dataloader = DataLoader(dataset=valset, num_workers=0, batch_size=1, shuffle=True)

    train_results_folder = OrderedDict()
    train_results_folder['psnr'] = []
    train_results_folder['ssim'] = []

    train_loss_fn, test_loss_fn = selections.loss_fun_select(args)
    model = selections.model_select(args, args.model_name).to(args.device)
    print('# trainable parameters: ', qis_utils.count_parameters(model))

    model_saved_name = selections.model_name_select(args, args.model_name)

    print('spat Model path: %s' % model_saved_name)

    #print(model)
    input = torch.ones((1,11,1,256,256)).to(args.device)
    flops = FlopCountAnalysis(model, input)
    print('total flops', flops.total())

    # Optimizer
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08,
                                                           verbose=False)

    val_psnr_best = 0

    iteration = 0

    if args.load_model_flag:
        state_dict_path = qis_utils.find_latest_checkpoint(args)
        checkpoint = torch.load(state_dict_path, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
        print('weights initialized with saved model at location: %s' % state_dict_path)
        if not args.start_over:
            iteration = checkpoint['iter']
            val_psnr_best = checkpoint['best_psnr']
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_lr = optimizer.param_groups[0]['lr']

            print('------------------------------------------------------------------------------')
            print("==> Resuming Training with learning rate:", new_lr)
            print('------------------------------------------------------------------------------')
            print('val psnr best: ', val_psnr_best)
            print('iter: ', iteration)

    if gpu_count > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(gpu_count)]).cuda()

    model.train()
    for epoch in range(args.start_epoch, args.total_epochs):
        start_ep = time.time()
        print('Epoch {%d + 1}/{%d}\n-------------------------------' % (epoch, args.total_epochs))

        size = len(t_dataloader.dataset)

        current_qis_gt_loss = 0
        current_out_gt_loss = 0
        current_qis_out_loss = 0
        fidx = (args.past_frames + args.future_frames + 1) // 2
        
        for batch, data in enumerate(t_dataloader):
            for param in model.parameters():
                param.grad = None
            qis_seq, gt_seq = data
            
            qis_seq = (qis_seq.to(torch.float32)).to(args.device)
            gt_seq = (gt_seq.to(torch.float32)).to(args.device)
            gt1_seq = gt_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            
            gt2_seq = F.interpolate(gt1_seq, size=(args.inp_ch, args.patch_size // 2, args.patch_size // 2),
                                    mode='trilinear', align_corners=False)
            gt3_seq = F.interpolate(gt1_seq, size=(args.inp_ch, args.patch_size // 4, args.patch_size // 4),
                                    mode='trilinear', align_corners=False)

            preden_seq, out1, out2, out3 = model(qis_seq)

            qis_seq = qis_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            
            qis_seq, gt_seq, gt1_seq, gt2_seq, gt3_seq, preden_seq, out1, out2, out3 = qis_seq.flatten(0, 1), gt_seq.flatten(0, 1), gt1_seq.flatten(0, 1), gt2_seq.flatten(0, 1), gt3_seq.flatten(0, 1), preden_seq.flatten(0, 1), out1.flatten(0, 1), out2.flatten(0, 1), out3.flatten(0, 1)
            
            qis_gt_loss = train_loss_fn(qis_seq, gt1_seq)
            
            out_gt_loss = 0.85 * train_loss_fn(out1, gt1_seq) + 0.1 * train_loss_fn(out2, gt2_seq) \
                          + 0.05 * train_loss_fn(out3, gt3_seq) + 0.2 * train_loss_fn(preden_seq, gt_seq)
            qis_out_loss = train_loss_fn(out1, qis_seq)
            
            out_gt_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            current_qis_gt_loss += qis_gt_loss.item()
            current_out_gt_loss += train_loss_fn(out1, gt1_seq).item()
            current_qis_out_loss += qis_out_loss.item()
            
            # Logging
            if batch % args.log_every == 0:
                current, lr = (batch + 1) * args.batch_size, qis_utils.get_lr(optimizer)
                print('[avg_qis_gt_loss: %.6f] [avg_out_gt_loss: %.6f] [avg_qis_out_loss: %.6f] [qis_gt_loss: %.6f] [out_gt_loss: %.6f] [qis_out_loss: %.6f] [%d / %d] [lr: %.6f]' % (
                    current_qis_gt_loss / (batch + 1), current_out_gt_loss / (batch + 1),
                    current_qis_out_loss / (batch + 1), qis_gt_loss.item(), train_loss_fn(out1, gt1_seq).item(), qis_out_loss.item(), current,
                    size * args.num_frames, lr))

            del qis_seq, preden_seq
            del out1, out2, out3
            del gt_seq, gt1_seq, gt2_seq, gt3_seq
            
            iteration += 1

            if iteration % args.save_period == 0:
                torch.save({'iter': iteration,
                            'best_psnr': val_psnr_best,
                            'state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(args.weights_dir, args.model_name + '_iter_%07d.pth' % iteration))

                val_psnr = validation(args, v_dataloader, model, iteration)
                
                # Save the best SNR model
                if val_psnr_best <= val_psnr:
                    val_psnr_best = val_psnr
                    best_epoch = epoch
                    torch.save(model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                               model_saved_name + '_best.pth')
                    print('best model saved at %s' % model_saved_name)
                    

                scheduler.step(val_psnr)
                torch.cuda.empty_cache()
                model.train()
                
        print('val psnr best %.6f' % val_psnr_best)
        # Logging
        time_per_epoch = (time.time() - start_ep) / 60
        print('Time per epoch %.3f mins' % time_per_epoch)

    time_tot = (time.time() - start_tot) / 60
    print('Total training time: %.4f mins\n' % time_tot)

    return model_saved_name, model


def validation(args, dataloader, model, iteration):
    model.eval()
    psnr = 0
    fidx = (args.past_frames + args.future_frames + 1) // 2
    size = len(dataloader.dataset)
    print(size)
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            qis_seq, gt_seq = data
            qis_seq = (qis_seq.to(torch.float32)).to(args.device)
            gt_seq = (gt_seq.to(torch.float32)).to(args.device)
            gt_seq = gt_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            
            _, out1, _, _ = model(qis_seq)

            qis_seq = qis_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            
            psnr += qis_utils.batch_psnr(out1.clamp(0.0, 1.0), gt_seq.clamp(0.0, 1.0), qis_seq.clamp(0.0, 1.0),
                                             data_range=1.0, plotdir=args.plotdir, iteration=iteration,
                                             visualize=args.visualize)
    print('psnr: %.3f' % (psnr / size))
    del qis_seq
    del gt_seq
    del out1
    
    return psnr / size


if __name__ == '__main__':
    # Get input arguments (e.g., config_id)
    parser = argparse.ArgumentParser(description='video denoising parameters')
    quiver_qis_input_args.quiver_training_args(parser)
    quiver_qis_input_args.sensor_args(parser)
    args = parser.parse_args()

    model_saved_name, model = main(args)
    print(model_saved_name)
