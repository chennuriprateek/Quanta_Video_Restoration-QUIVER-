import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile
# from einops import rearrange
import argparse, os, math
from archs import conv1x1, conv3x3, conv5x5, conv5x5x5, actFunc, DCNv2PackFlowGuided, Conv2dLayer


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4':  # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode,
                               align_corners=align_corners)

        return output


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, load_spynet_weights=False, return_levels=[3]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(4)])
        if load_path:
            if load_spynet_weights:
                if not os.path.exists(load_path):
                    import requests
                    url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                    r = requests.get(url, allow_redirects=True)
                    print(f'downloading SpyNet pretrained model from {url}')
                    os.makedirs(os.path.dirname(load_path), exist_ok=True)
                    open(load_path, 'wb').write(r.content)
                print("loading spynet weights...")
                self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.449]).view(1, 1, 1, 1))
        self.register_buffer('std', torch.Tensor([0.226]).view(1, 1, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(3):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2 ** (3 - level)  # level=3 (scale=1), level=2 (scale=2), level=1 (scale=4)
                flow_out = F.interpolate(input=flow, size=(h // scale, w // scale), mode='bilinear',
                                         align_corners=False)
                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size=[3, 3], reduction=16, activation='gelu'):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size[0], padding=kernel_size[0] // 2, bias=True))
        modules_body.append(actFunc(activation))
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size[1], padding=kernel_size[1] // 2, bias=True))

        self.CA = CALayer(n_feat, reduction)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


# RDB fusion module
class RDB_shrink(nn.Module):
    def __init__(self, in_channels, growthRate, output_channels, num_layer, activation='relu'):
        super(RDB_shrink, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.shrink = conv3x3(in_channels, output_channels, stride=1)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.shrink(x)
        return out


# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out


# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, output_channels, num_layer, activation='relu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, output_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out


# RDB-based RNN cell
class RDBCell(nn.Module):
    def __init__(self, para):
        super(RDBCell, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_B01 = conv5x5(para.inp_ch, self.n_feats, stride=1)
        self.F_B01_fuse = conv1x1(2 * self.n_feats, 4 * self.n_feats)
        self.F_B02 = conv3x3(para.inp_ch, self.n_feats, stride=1)
        self.F_B03 = conv3x3(para.inp_ch, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(in_channels=4 * self.n_feats, growthRate=self.n_feats, output_channels=4 * self.n_feats,
                           num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=6 * self.n_feats, growthRate=int(self.n_feats * 3 / 2),
                           output_channels=4 * self.n_feats, num_layer=3, activation=self.activation)
        self.F_R = nn.Sequential(
            RDB_shrink(in_channels=8 * self.n_feats, growthRate=self.n_feats,
                       output_channels=4 * self.n_feats, num_layer=3, activation=self.activation),
            RDNet(in_channels=4 * self.n_feats, growthRate=2 * self.n_feats,
                  num_layer=3, num_blocks=self.n_blocks - 1, activation=self.activation)
        )
        # F_h: hidden state part
        self.F_h = nn.Sequential(
            conv3x3(4 * self.n_feats, 2 * self.n_feats),
            RDB(in_channels=2 * self.n_feats, growthRate=2 * self.n_feats, num_layer=3, activation=self.activation),
            conv3x3(2 * self.n_feats, 2 * self.n_feats)
        )

    def forward(self, xs, x_feats, s_last):
        '''
        input: size 256 256
        0: torch.Size([2, 64, 256, 256])
        1: torch.Size([2, 64, 128, 128])
        2: torch.Size([2, 96, 64, 64])
        3: torch.Size([2, 96, 64, 64])
        '''
        x1, x2, x3 = xs
        b, c, h, w = x1.shape

        x_feat1, x_feat2, x_feat3 = x_feats

        out1 = self.F_B01(x1)
        out1 = self.F_B01_fuse(torch.cat([out1, x_feat1], 1))
        inp2 = self.F_B02(x2)
        inp3 = self.F_B03(x3)
        # in: 2 n_feats, out: 2 * n_feats
        out2 = self.F_B1(out1)
        # in: 2+1+1 n_feats, out: 3 * n_feats
        out3 = self.F_B2(torch.cat([out2, inp2, x_feat2], 1))
        out3 = torch.cat([out3, inp3, x_feat3, s_last], dim=1)
        # in: 3+1+1+2 n_feats, out: 5 * n_feats
        out3 = self.F_R(out3)
        # in: 5 n_feats, out: 2 * n_feats
        s = self.F_h(out3)
        # print(out1.shape, out2.shape, out3.shape)
        return out1, out2, out3, s


class self_att(nn.Module):
    def __init__(self, dim):
        super(self_att, self).__init__()
        self.dim = dim
        self.get_qkv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
                                     nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.get_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.dim, h * w)
        k = k.view(b, self.dim, h * w)
        v = v.view(b, self.dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = out.view(b, self.dim, h, w)
        out = self.project_out(out)
        return out

class spatial_att(nn.Module):
    def __init__(self, para):
        super(spatial_att, self).__init__()
        self.para = para
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + self.num_fb + 1
        self.fidx = self.related_f//2
        
        self.conv = nn.Conv2d(self.related_f * 4 * self.n_feats, self.related_f * 4 * self.n_feats, kernel_size=3, bias=False, padding=3 // 2)
        self.spat_att = self_att(self.related_f * 4 * self.n_feats)
        self.conv1x1 = nn.Conv2d(self.related_f * 4 * self.n_feats, 4 * self.n_feats, kernel_size=1, bias=False)
    
    def forward(self, x):
        x.append(x.pop(self.fidx))
        x = self.conv(torch.cat(x, dim=1))
        att = self.spat_att(x)
        return self.conv1x1(att + x)

# Reconstructor (prateek)
class Reconstructor(nn.Module):
    def __init__(self, para, in_channels):
        super(Reconstructor, self).__init__()
        self.para = para
        self.in_channels = in_channels
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        
        self.conv1x1 = conv1x1(in_channels, 8 * self.n_feats)
        self.cab = CAB(8 * self.n_feats, kernel_size=[1, 3])
        self.flast = nn.Sequential(
            conv3x3(8 * self.n_feats, 8 * self.n_feats),
            actFunc(para.activation),
            conv3x3(8 * self.n_feats, 6 * self.n_feats),
            actFunc(para.activation),
            conv3x3(6 * self.n_feats, 4 * self.n_feats),
            actFunc(para.activation),
            nn.ConvTranspose2d(4 * self.n_feats, 4 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        )
        self.out = nn.Sequential(
        conv3x3(8 * self.n_feats, 6 * self.n_feats),
        actFunc(para.activation),
        conv3x3(6 * self.n_feats, 4 * self.n_feats),
        actFunc(para.activation),
        conv3x3(4 * self.n_feats, 2 * self.n_feats),
        actFunc(para.activation),
        conv3x3(2 * self.n_feats, self.n_feats),
        conv3x3(self.n_feats, para.inp_ch)
        )
        self.upconv_out = nn.ConvTranspose2d(para.inp_ch, para.inp_ch, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)

    def forward(self, h, flast, residual):
        h = self.conv1x1(torch.cat([h, flast], dim=1))
        attn = self.cab(h)
        flast = self.flast(attn)
        out = self.out(attn)
        out = out + residual
        residual = self.upconv_out(out)
        return out, flast, residual


class FeatureExtractor(nn.Module):
    def __init__(self, para):
        super(FeatureExtractor, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.f1 = conv5x5x5(para.inp_ch, self.n_feats, stride=1)
        self.f2 = conv5x5x5(para.inp_ch, self.n_feats, stride=(1, 2, 2))
        self.f3 = conv5x5x5(para.inp_ch, self.n_feats, stride=(1, 2, 2))

    def forward(self, x):
        # channel: 10, 4, 2 * n_feat
        b, t, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        out1 = self.f1(x).permute(0, 2, 1, 3, 4)
        out2 = self.f2(x).permute(0, 2, 1, 3, 4)
        out3 = self.f3(F.interpolate(x, size=(t, h // 2, w // 2), mode='trilinear', align_corners=False))
        return out1, out2, out3.permute(0, 2, 1, 3, 4)


class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x

class predenoise(nn.Module):
    def __init__(self,para):
        super(predenoise, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.sfe1 = nn.Conv2d(para.inp_ch, self.n_feats, kernel_size=1)
        self.sfe2 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=3 // 2)
        self.num_rdbs = 5
        
        self.rdbs = nn.ModuleList([RDB(self.n_feats, growthRate=64, num_layer=3, activation='relu')]*self.num_rdbs)
        self.gff = nn.Sequential(
            nn.Conv2d(self.n_feats * self.num_rdbs, self.n_feats, kernel_size=1),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=3 // 2)
        )
        self.out = nn.Conv2d(self.n_feats, para.inp_ch, kernel_size=3, padding=3 // 2)
    
    def forward(self, x):
        x1 = x.clone()
        x = self.sfe1(x)
        x2 = x.clone()
        x = self.sfe2(x)
        rdb_feats = []
        for i in range(self.num_rdbs):
            x = self.rdbs[i](x)
            rdb_feats.append(x)
        x = self.gff(torch.cat(rdb_feats, dim=1)) + x2
        x = self.out(x) + x1
        return x
        
class QUIVER(nn.Module):
    def __init__(self, para):
        super(QUIVER, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + self.num_fb + 1
        self.fidx = self.related_f//2
        self.ds_ratio = 4
        self.predenoise = predenoise(para)
        self.spynet = SpyNet(para.spynet_path, para.load_spynet_weights, [1, 2, 3])
        self.device = torch.device('cuda')
        self.extractor = FeatureExtractor(para)
        self.preden_extractor = FeatureExtractor(para)
        #changes made here -- Nov 3rd
        self.alignfuse_1 = Mlp_GEGLU(para.n_features * 2, para.n_features * 2, para.n_features)
        self.alignfuse_2 = Mlp_GEGLU(para.n_features * 2, para.n_features * 2, para.n_features)
        self.alignfuse_3 = Mlp_GEGLU(para.n_features * 2, para.n_features * 2, para.n_features)
        # self.pre_align = RD_CA_cell(para)
        self.forward_cell = RDBCell(para)
        #self.backward_cell = RDBCell(para)
        self.spatial_att = spatial_att(para)
        self.recons3 = Reconstructor(para, 8 * self.n_feats)
        self.recons12 = Reconstructor(para, 8 * self.n_feats)
        self.pa_deform = DCNv2PackFlowGuided(self.n_feats, self.n_feats, 3, padding=1, deformable_groups=16,
                                             max_residue_magnitude=10, pa_frames=2)

    def forward(self, x):
        out1, out2, out3, hf1, hf2, hf3 = [], [], [], [], [], []
        batch_size, frames, channels, height, width = x.shape
        x_down2 = F.interpolate(x, size=(channels, height // 2, width // 2), mode='trilinear', align_corners=False)
        x_down4 = F.interpolate(x, size=(channels, height // 4, width // 4), mode='trilinear', align_corners=False)
        
        preden = x.clone()
        for b in range(x.shape[0]):
            preden[b,...] = self.predenoise(x[b,...])
            
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        
        flows = self.get_flow(preden)
        x_feat_1, x_feat_2, x_feat_3 = self.extractor(x)
        preden_feat_1, preden_feat_2, preden_feat_3 = self.preden_extractor(preden)
        
        x_warped_1 = self.get_aligned_feature_2frames(x_feat_1, flows[0])
        x_warped_2 = self.get_aligned_feature_2frames(x_feat_2, flows[1])
        x_warped_3 = self.get_aligned_feature_2frames(x_feat_3, flows[2])
        preden_warped_1 = self.get_aligned_feature_2frames(preden_feat_1, flows[0])
        preden_warped_2 = self.get_aligned_feature_2frames(preden_feat_2, flows[1])
        preden_warped_3 = self.get_aligned_feature_2frames(preden_feat_3, flows[2])
        
        #changes made here -- Nov 3rd
        x_warped_1 = self.alignfuse_1(torch.cat([preden_warped_1, x_warped_1], 2).permute(0, 1, 3, 4, 2)).permute(0, 1, 4,
                                                                                                             2, 3)
        x_warped_2 = self.alignfuse_2(torch.cat([preden_warped_2, x_warped_2], 2).permute(0, 1, 3, 4, 2)).permute(0, 1, 4,
                                                                                                             2, 3)
        x_warped_3 = self.alignfuse_3(torch.cat([preden_warped_3, x_warped_3], 2).permute(0, 1, 3, 4, 2)).permute(0, 1, 4,
                                                                                                             2, 3)
        
        # forward h structure: (batch_size, channel, height, width)
        s = torch.zeros(batch_size, 2 * self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            x_feats = [x_warped_1[:, i, ...], x_warped_2[:, i, ...], x_warped_3[:, i, ...]]
            xs = [x[:, i, ...], x_down2[:, i, ...], x_down4[:, i, ...]]
            h1, h2, h3, s = self.forward_cell(xs, x_feats, s)
            hf1.append(h1)
            hf2.append(h2)
            hf3.append(h3)
        
        for i in range(self.num_fb, frames - self.num_ff):
            out = self.spatial_att(hf3[i - self.num_fb:i + self.num_ff + 1])
            flast = torch.zeros(batch_size, 4 * self.n_feats, height//4, width//4).to(self.device)
            residual = torch.zeros(batch_size, self.para.inp_ch, height//4, width//4).to(self.device)
            
            o3, flast, residual = self.recons3(out, flast, residual)
            o2, flast, residual = self.recons12(hf2[i], flast, residual)
            o1, _, _ = self.recons12(hf1[i], flast, residual)
            out1.append(o1.unsqueeze(dim=1))
            out2.append(o2.unsqueeze(dim=1))
            out3.append(o3.unsqueeze(dim=1))
        
        return preden, torch.cat(out1, dim=1), torch.cat(out2, dim=1), torch.cat(out3, dim=1)

    def get_flow(self, x, levels=3):
        '''Get flow between frames t and t+1 from x.'''

        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # forward
        flows_forward = self.spynet(x_2, x_1)
        flows_forward = [flow.view(b, n - 1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                         zip(flows_forward, range(levels))]
        return flows_forward

    def get_aligned_feature_2frames(self, x, flows_forward):
        '''Parallel feature warping for 2 frames.'''

        # backward
        n = x.size(1)

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_forward.append(self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

        return torch.stack(x_forward, 1)

def feed(model, iter_samples):
    inputs = iter_samples
    outputs = model(inputs)
    return outputs
