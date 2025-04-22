import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import numpy as np
from rwkv2 import RWKV, RWKVConfig
from embd import DataEmbeddingVIO

import torchvision.models as models
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        ).to(torch.float)
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        ).to(torch.float)

class InertialEncoderWithResidual(nn.Module):
    def __init__(self, opt):
        super(InertialEncoderWithResidual, self).__init__()

        # Initial convolution layer
        self.initial_conv = nn.Conv1d(6, 64, kernel_size=3, padding=1).to(torch.float)
        self.initial_bn = nn.BatchNorm1d(64)
        self.initial_relu = nn.LeakyReLU(0.1, inplace=False)
        self.initial_dropout = nn.Dropout(opt.imu_dropout)

        # First residual block (using two 3x3 convs)
        self.res1_conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.res1_bn1 = nn.BatchNorm1d(64)
        self.res1_relu1 = nn.LeakyReLU(0.1, inplace=False)
        self.res1_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.res1_bn2 = nn.BatchNorm1d(64)
        self.res1_relu2 = nn.LeakyReLU(0.1, inplace=False)
        self.res1_dropout = nn.Dropout(opt.imu_dropout)

        # Second residual block (using two 3x3 convs)
        self.res2_conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.res2_bn1 = nn.BatchNorm1d(128)
        self.res2_relu1 = nn.LeakyReLU(0.1, inplace=False)
        self.res2_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.res2_bn2 = nn.BatchNorm1d(128)
        self.res2_relu2 = nn.LeakyReLU(0.1, inplace=False)
        self.res2_dropout = nn.Dropout(opt.imu_dropout)
        self.res2_project = nn.Conv1d(64, 128, kernel_size=1)

        # Third residual block (using two 3x3 convs)
        self.res3_conv1 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.res3_bn1 = nn.BatchNorm1d(256)
        self.res3_relu1 = nn.LeakyReLU(0.1, inplace=False)
        self.res3_conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.res3_bn2 = nn.BatchNorm1d(256)
        self.res3_relu2 = nn.LeakyReLU(0.1, inplace=False)
        self.res3_dropout = nn.Dropout(opt.imu_dropout)
        self.res3_project = nn.Conv1d(128, 256, kernel_size=1)

        # Projection layer
        self.proj = nn.Linear(256 * 11, opt.i_f_len).to(torch.float)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))
        x = x.permute(0, 2, 1)  # Permute to match Conv1d input shape [batch_size * seq_len, length, channels]

        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.initial_dropout(x)
        initial_output = x  # Save initial convolution output for residual connection

        # First residual block
        residual = initial_output
        x = self.res1_conv1(x)  # [batch_size * seq_len, 64, length]
        x = self.res1_bn1(x)
        x = self.res1_relu1(x)
        x = self.res1_conv2(x)  # [batch_size * seq_len, 64, length]
        x = self.res1_bn2(x)
        x = self.res1_relu2(x)
        x = self.res1_dropout(x)
        x += residual  # Add residual connection

        # Second residual block
        residual = x
        x = self.res2_conv1(x)
        x = self.res2_bn1(x)
        x = self.res2_relu1(x)
        x = self.res2_conv2(x)
        x = self.res2_bn2(x)
        x = self.res2_relu2(x)
        x = self.res2_dropout(x)
        # Adjust residual to match new shape by projecting channels
        residual = self.res2_project(residual)  # Project residual to match x's channel dimension
        x += residual  # Add residual connection

        # Third residual block
        residual = x
        x = self.res3_conv1(x)
        x = self.res3_bn1(x)
        x = self.res3_relu1(x)
        x = self.res3_conv2(x)
        x = self.res3_bn2(x)
        x = self.res3_relu2(x)
        x = self.res3_dropout(x)
        # Adjust residual to match new shape by projecting channels
        residual = self.res3_project(residual)  # Project residual to match x's channel dimension
        x += residual  # Add residual connection

        # Flatten and project
        x = x.view(x.shape[0], -1)
        out = self.proj(x)
        return out.view(batch_size, seq_len, 256)

def new_encoder_initialization(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1).to(torch.float),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout)).to(torch.float)
        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len).to(torch.float)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        print(x.shape)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))
        x = self.encoder_conv(x.permute(0, 2, 1))
        out = self.proj(x.view(x.shape[0], -1))
        return out.view(batch_size, seq_len, 256)


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)
        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len).to(torch.float)
        self.inertial_encoder = InertialEncoderWithResidual(opt)
        self.inertial_encoder2 = InertialEncoderWithResidual(opt)

    def forward(self, img, imu):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)
        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        # v_depth = self.depth(v)
        v = self.encode_image(v)
        # print(f"v feature: {v.shape}")
        v = v.view(batch_size, seq_len, -1).to(torch.float)
        v = self.visual_head(v)

        # IMU CNN
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        print(imu.shape)
        imu1 = self.inertial_encoder(imu)
        imu2 = self.inertial_encoder2(imu)
        imu = torch.cat((imu1, imu2), -1)
        return v, imu

    def encode_image(self, x):
        x = x.to(torch.float)
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

class Fusion_module(nn.Module):
    def __init__(self, opt):
        super(Fusion_module, self).__init__()
        self.fuse_method = opt.fuse_method
        self.v_len = 512  # 视觉特征长度
        self.i_len = 256  # IMU 特征长度
        self.f_len = 1024  # 最终对齐后的特征长度
        # 使用自注意力机制来生成动态权重
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len)).to(torch.float)
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len)).to(torch.float)
        # self.alpha = nn.Parameter(torch.tensor(0.2))  # 初始值为 0.5
        # self.net = nn.Sequential(
        #     nn.Linear(self.v_len, 2 * self.v_len)).to(torch.float)

    def forward(self, v, i):
        if self.fuse_method == 'dynamic_cat':
            weights = self.net(v)  # [batch_size, seq_len, 2 * f_len]
            weights = weights.view(v.shape[0], v.shape[1], self.v_len, 2)
            # 使用软掩码和硬掩码结合
            soft_mask = F.softmax(weights, dim=-1)  # 软掩码
            hard_mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)  # 硬掩码
            # 融合软硬掩码
            mixed_mask = self.alpha * hard_mask + (1 - self.alpha) * soft_mask
            v = v * mixed_mask[:, :, :, 0]
            # 返回加权特征
            return torch.cat((v, i), -1)    # 使用混合掩码

        elif self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]

class Pose_RWKV_TS(nn.Module):
    def __init__(self, opt):
        super(Pose_RWKV_TS, self).__init__()
        rwkv_config = RWKVConfig(
            n_layer=opt.n_layer,
            n_head=opt.n_heads,
            n_embd=opt.n_embd,
            dropout=opt.rwkv_dropout_out
        )
        self.enc_emb = DataEmbeddingVIO(feature_dim=1024, d_model=512)
        self.rwkv = RWKV(rwkv_config)
        self.fuse = Fusion_module(opt)
        self.rnn_drop_out = nn.Dropout(0.04)
        self.regressor = nn.Sequential(
            nn.Linear(opt.n_embd, 128).to(torch.float),
            nn.LeakyReLU(0.1, inplace=True).to(torch.float),
            nn.Linear(128, 6).to(torch.float),
        )
    def forward(self, fv, fi):
        # fi = torch.cat((fi, fi2), -1)
        fused = self.fuse(fv, fi)   #(bs, seq-1, 768)
        fused = self.enc_emb(fused)
        out = self.rwkv(fused)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)
        return pose


class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()
        self.Feature_net = Encoder(opt)
        self.Pose_net = Pose_RWKV_TS(opt)
        self.opt = opt
        for name, module in self.named_modules():
            if "Feature_net.inertial_encoder2" and "MobileViTv2Attention" not in name:
                initialization(module)
        new_encoder_initialization(self.Feature_net.inertial_encoder2)
        initialization(self)
    def forward(self, img, imu):
        img = img.to(torch.float)
        imu = imu.to(torch.float)
        fv, fi = self.Feature_net(img, imu)
        poses = self.Pose_net(fv, fi)
        return poses

def initialization(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
