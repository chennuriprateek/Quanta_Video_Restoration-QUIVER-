import os
import glob

from random import choices  # requires Python >= 3.6
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import random
from skimage.metrics import structural_similarity as SSIM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def find_latest_checkpoint(args):
    if 'disc' in args.model_name:
        gen_path = os.path.join(args.weights_dir, args.model_name + '_' + args.gen_name)
        disc_path = os.path.join(args.weights_dir, args.model_name + '_' + args.disc_name)
        gen_state_dict = sorted(glob.glob(gen_path + '_iter_*.pth'), reverse = True)[0]
        disc_state_dict = sorted(glob.glob(disc_path + '_iter_*.pth'), reverse=True)[0]
        return gen_state_dict, disc_state_dict
    else:
        path = os.path.join(args.weights_dir, args.model_name)
        model_state_dict = sorted(glob.glob(path + '_iter_*.pth'), reverse = True)[0]
        return model_state_dict

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def frames_extraction(video_path, frames_no, start_frame = None, downsample = None):
    vid = cv2.VideoCapture(video_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames<15:
        print('*************************')
        print(video_path)
        print('*************************')
    if start_frame == None and total_frames>15:
        start_frame = random.randint(0, total_frames - 1 - frames_no)
    else:
        start_frame = start_frame
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    seq_list = []
    for _ in range(frames_no):
        _, img = vid.read()
        seq_list.append(img)
    
    for f in range(len(seq_list)):
        img = seq_list[f]    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if downsample != None:
            c = 1024
            r = 512
            output_shape = (c + (c % 4), r + (r % 4))
        else:
            output_shape = (img.shape[1] + (img.shape[1] % 4), img.shape[0] + (img.shape[0] % 4))
        img = cv2.resize(img, output_shape, interpolation=cv2.INTER_CUBIC)
        seq_list[f] = img
    
    seq = np.stack(seq_list, axis=0)
    
    return seq
    

def open_sequence(files, gray_mode=False, expand_if_needed=False):
    r""" Opens a sequence of images and expands it to even sizes if necesary
	Args:
		fpath: string, path to image sequence
		gray_mode: boolean, True indicating if images is to be open are in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
		max_num_fr: maximum number of frames to load
	Returns:
		seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
    :param expand_if_needed:
    :param gray_mode:
    :param files:
	"""

    seq_list = []
    for fpath in files:
        img, expanded_h, expanded_w = open_image(fpath,
                                                 gray_mode=gray_mode,
                                                 expand_if_needed=expand_if_needed,
                                                 expand_axis0=False,
                                                 normalize_data=False)
        seq_list.append(img)
        seq = np.stack(seq_list, axis=0)

    # print("\tLoaded sequence with starting file", files[0])
    return seq, expanded_h, expanded_w


def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=False):
    r""" Opens an image and expands it if necesary
	Args:
		fpath: string, path of image file
		gray_mode: boolean, True indicating if image is to be open
			in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
	Returns:
		img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
			if expand_axis0=False, the output will have a shape CxHxW.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
    :param fpath:
    :param gray_mode:
    :param expand_if_needed:
    :param expand_axis0:
    :param normalize_data:
	"""
    if not gray_mode:
        # Open image as a CxHxW torch.Tensor
        img = cv2.imread(fpath)
        # from HxWxC to CxHxW, RGB image
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        # Handle odd sizes
        output_shape = (img.shape[1] + (img.shape[1] % 4), img.shape[0] + (img.shape[0] % 4))
        img = cv2.resize(img, output_shape, interpolation=cv2.INTER_CUBIC)

    if expand_axis0:
        img = np.expand_dims(img, 0)

    expanded_h = False
    expanded_w = False

    if normalize_data:
        img = normalize(img)
    return img, expanded_h, expanded_w    


def sensor_image_simulation(avg_PPP, photon_flux, QE, theta_dark, sigma_read, N, Nbits, gain):
    min_val = 0
    max_val = 2 ** Nbits - 1
    theta = photon_flux * (avg_PPP / (np.mean(photon_flux) + 0.0001))
    lam = ((QE * theta) + theta_dark) / N

    m, n, c = theta.shape
    img_out = np.zeros((m, n, c))
    
    for i in range(N):
        tmp = np.random.poisson(lam=lam, size=(m, n, c))
        tmp = tmp + np.random.normal(loc=0, scale=sigma_read, size=(m, n, c))
        tmp = np.round(tmp*gain)
        tmp = np.clip(tmp, min_val, max_val)
        img_out = img_out + tmp

    img_out = img_out / N
    return img_out


def normalize(data, max_value=255.):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]
	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
    return np.float32(data / max_value)


def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
    r"""Converts a torch.autograd.Variable to an OpenCV image
	Args:
		invar: a torch.autograd.Variable
		conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
	Returns:
		a HxWxC uint8 image
	"""
    assert torch.max(invar) <= 1.0

    size4 = len(invar.size()) == 4
    if size4:
        nchannels = invar.size()[1]
    else:
        nchannels = invar.size()[0]

    if nchannels == 1:
        if size4:
            res = invar.data.cpu().numpy()[0, 0, :]
        else:
            res = invar.data.cpu().numpy()[0, :]
        res = (res * 255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        if size4:
            res = invar.data.cpu().numpy()[0]
        else:
            res = invar.data.cpu().numpy()
        res = res.transpose(1, 2, 0)
        res = (res * 255.).clip(0, 255).astype(np.uint8)
        if conv_rgb_to_bgr:
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    else:
        raise Exception('Number of color channels not supported')
    return res


def compare_psnr(img1, img2, data_range=255.0):
    return 10. * ((data_range ** 2) / ((img1 - img2) ** 2).mean()).log10()


def batch_ssim(img, imclean, data_range=1.0):
    ssim_val = 0
    for b in range(img.shape[0]):
        for i in range(img.shape[1]):
            ssim_val += SSIM(imclean[b, i, 0, :, :].detach().cpu().numpy(), img[b, i, 0, :, :].detach().cpu().numpy(), data_range=data_range)
    print('ssim: ', ssim_val/(img.shape[0]*img.shape[1])) 
    return ssim_val   


def batch_psnr(img, imclean, inp, data_range, plotdir, iteration, visualize = False):
    r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
    psnr = 0
    assert img.shape == imclean.shape
    assert img.shape == inp.shape
    
    for b in range(img.shape[0]):
        for i in range(img.shape[1]):
            psnr += compare_psnr(imclean[b, i, :, :, :], img[b, i, :, :, :],
                                 data_range=data_range)
    if visualize:
        frame = random.randint(0, img.shape[1] - 1)

        for b in range(img.shape[0]):
            out_seq = img[b, frame, ...].detach().cpu().numpy()
            qis_seq = inp[b, frame, ...].detach().cpu().numpy()
            gt_seq = imclean[b, frame, ...].detach().cpu().numpy()
            out_seq, gt_seq, qis_seq = out_seq.transpose(1, 2, 0), gt_seq.transpose(1, 2, 0), qis_seq.transpose(1,
                                                                                                                2,
                                                                                                                0)
            out_seq, gt_seq, qis_seq = np.clip(out_seq, 0, 1.0) * 255.0, np.clip(gt_seq, 0, 1.0) * 255.0, np.clip(
                qis_seq,
                0,
                1.0) * 255.0
            in_error = np.abs(qis_seq - gt_seq)
            out_error = np.abs(out_seq - gt_seq)
            in_out_error = np.abs(out_seq - qis_seq)
            out_seq, gt_seq, qis_seq, in_error, out_error, in_out_error = out_seq.astype(np.uint8), gt_seq.astype(
                np.uint8), \
                qis_seq.astype(np.uint8), in_error.astype(np.uint8), out_error.astype(
                np.uint8), in_out_error.astype(
                np.uint8)

            fig = plt.figure(figsize=(14, 5.5))

            plt.subplot(2, 3, 2)
            plt.imshow(qis_seq[...,0], cmap='gray')
            plt.axis('off')
            plt.title('QIS input')

            plt.subplot(2, 3, 1)
            plt.imshow(gt_seq[...,0], cmap='gray')
            plt.axis('off')
            plt.title('GT')

            plt.subplot(2, 3, 3)
            plt.imshow(out_seq[...,0], cmap='gray')
            plt.axis('off')
            plt.title('denoised o/p')

            plt.subplot(2, 3, 4)
            plt.imshow(in_out_error[...,0], vmin=np.min(in_out_error),
                       vmax=np.max(in_out_error), cmap='gray')
            plt.axis('off')
            plt.colorbar()
            plt.title('abs(QIS - out) %0.5f' % (np.mean(in_out_error)))

            plt.subplot(2, 3, 5)
            plt.imshow(in_error[...,0], vmin=np.concatenate([in_error, out_error]).ravel().min(),
                       vmax=np.concatenate([in_error, out_error]).ravel().max(), cmap='gray')
            plt.axis('off')
            plt.title('abs(QIS - GT) %0.5f' % (np.mean(in_error)))

            plt.subplot(2, 3, 6)
            plt.imshow(out_error[...,0], vmin=np.concatenate([in_error, out_error]).ravel().min(),
                       vmax=np.concatenate([in_error, out_error]).ravel().max(), cmap='gray')
            plt.axis('off')
            plt.title('abs(out - GT) %0.5f' % (np.mean(out_error)))
            plt.colorbar()
            plt.show()

            fig.savefig(os.path.join(plotdir, str(iteration) + '_%05d.png' % b))
            plt.close()
    
    print(psnr/(img.shape[0]*img.shape[1]))
    return psnr/(img.shape[0]*img.shape[1])


def adjust_learning_rate(lr_in, optimizer, epoch, args):
    """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
    lr = lr_in * (args.LR_factor ** (epoch // args.patience))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
