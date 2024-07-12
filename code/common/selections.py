import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'quiver')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'emvd')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'flornn')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'memdeblur')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rvrt')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spk2imgnet')))

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torch.nn.init as init


def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)
    

def model_select(args, model_name):
    if model_name == 'quiver':
        import quiver_qis_model
        model = quiver_qis_model.QUIVER(args)
        print('quiver model loaded from quiver_qis_model')
        #weights_init(model, init_type=args.init_type, init_gain=args.init_gain)
        return model
    elif model_name == 'rvrt':
        import rvrt_qis_model
        model = rvrt_qis_model.RVRT(args)
        print('rvrt model loaded from rvrt_qis_model')
        #weights_init(model, init_type=args.init_type, init_gain=args.init_gain)
        return model
    elif model_name == 'memdeblur':
        import memdeblur_qis_model
        model = memdeblur_qis_model.MEMDEBLUR(args)
        print('memdeblur model loaded from memdeblur_qis_model')
        #weights_init(model, init_type=args.init_type, init_gain=args.init_gain)
        return model
    elif model_name == 'flornn':
        import flornn_qis_model
        model = flornn_qis_model.FloRNN(args.inp_ch, num_resblocks=args.num_resblocks, num_channels=64, forward_count=args.forward_count, border_ratio=args.border_ratio)
        print('flornn model loaded from flornn_qis_model')
        return model
    elif model_name == 'emvd':
        import emvd_qis_model
        model = emvd_qis_model.MainDenoise()
        print('MainDenoise model loaded from emvd_qis_model')
        return model
    elif model_name == 'spk2imgnet':
        import spk2imgnet_qis_model
        model = spk2imgnet_qis_model.SpikeNet(args.inp_ch, args.n_features, out_channels=1)
        print('SpikeNet model loaded from spk2imgnet_qis_model')
        return model
    else:
        print('Invalid model, exiting...')
        exit()


def model_name_select(args, model_name):
    j = os.path.join

    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)

    return j(args.weights_dir, model_name)


def loss_fun_select(args):
    if args.loss_fun_name == 'MSE':
        return nn.MSELoss(reduction='mean'), nn.L1Loss(reduction='mean')
    elif args.loss_fun_name == 'L1':
        return nn.L1Loss(reduction='mean'), nn.L1Loss(reduction='mean')
    elif args.loss_fun_name == 'char':
        return CharbonnierLoss(), nn.L1Loss(reduction='mean')
    elif args.loss_fun_name == 'L1_grad':
        return L1_grad_loss(args.device), nn.L1Loss(reduction='mean')
    elif args.loss_fun_name == 'L1_grad_pl':
        print('Cost function includes perceptual loss...')
        return L1_grad_perceptual_loss(args.device), nn.L1Loss(reduction='mean')
    else:
        print('Invalid loss function, exiting...')
        exit()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class L1_grad_perceptual_loss(nn.Module):
    def __init__(self, device):
        super(L1_grad_perceptual_loss, self).__init__()
        self.L1_grad = L1_grad_loss(device)
        self.pl = VGGPerceptualLoss(device)

    def forward(self, inp, target):
        return self.L1_grad(inp, target) + 0.12*self.pl(inp, target)


class L1_grad_loss(nn.Module):
    def __init__(self, device):
        super(L1_grad_loss, self).__init__()
        a = torch.Tensor([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]]).to(device)
        self.a = a.view((1, 1, 3, 3))

        b = torch.Tensor([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]]).to(device)
        self.b = b.view((1, 1, 3, 3))
        self.L1loss = nn.L1Loss(reduction='mean')

    def forward(self, x, y):
        batch = x.shape[0]
        G_x_x = []
        G_x_y = []
        G_y_x = []
        G_y_y = []
        for i in range(batch):
            G_x_x.append(F.conv2d(x[i:i+1, ...], self.a)[None, ...])
            G_x_y.append(F.conv2d(x[i:i+1, ...], self.b)[None, ...])
            G_y_x.append(F.conv2d(y[i:i+1, ...], self.a)[None, ...])
            G_y_y.append(F.conv2d(y[i:i+1, ...], self.b)[None, ...])

        G_x_x = torch.cat(G_x_x, dim=0)
        G_x_y = torch.cat(G_x_y, dim=0)
        G_y_x = torch.cat(G_y_x, dim=0)
        G_y_y = torch.cat(G_y_y, dim=0)

        return self.L1loss(x, y) + 0.5 * (self.L1loss(G_x_x, G_y_x) + self.L1loss(G_x_y, G_y_y))


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.blocks = self.blocks.to(device)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
