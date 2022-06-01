# import torch package
from signal import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        if inchannel == outchannel:
          self.out = nn.Sequential(
              nn.Conv1d(inchannel, outchannel//2, kernel_size=1, stride=stride, padding=1, bias=False),
              nn.BatchNorm1d(outchannel),
              nn.ReLU(inplace=True),
              nn.Conv1d(outchannel//2, outchannel//2, kernel_size=3, stride=1, padding=1, bias=False),
              nn.BatchNorm1d(outchannel),
              nn.ReLU(inplace=True),
              nn.Conv1d(outchannel//2, outchannel, kernel_size=1, stride=1, padding=1, bias=False),
              nn.BatchNorm1d(outchannel)
          )
        # self.shortcut = nn.Sequential()
        self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )
        if stride != 1 or inchannel != outchannel:
            #shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )
            
    def forward(self, x):
        # print(x.shape)
        out = self.left(x)
        # out = self.test(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, n_channel):
      super(ChannelAttention, self).__init__()
      self.linear = nn.Sequential(
        nn.Linear(n_channel, n_channel//2),
        nn.ReLU(inplace=True),
        nn.Linear(n_channel//2, n_channel),
        nn.ReLU(inplace=True),
        nn.Linear(n_channel, n_channel)
    )
                                
      self.adapt_avg = nn.AdaptiveAvgPool1d(1)
      self.adapt_max = nn.AdaptiveMaxPool1d(1)
  
    def forward(self, x):
      size = x.size()
      avg, max = self.adapt_avg(x).view(size[0], -1), self.adapt_max(x).view(size[0], -1)
      # x_cat = torch.cat((avg, max), -1)
      x_avg = self.linear(avg)
      x_max = self.linear(max)
      x_att = torch.sigmoid(x_avg + x_max).view(size[0], size[1], 1)
      x = x_att * x
      return x

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, init_threshold, signal_shape):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.signal_shape = signal_shape
        # self.k1= nn.Parameter(torch.tensor(k1))
        # self.k2= nn.Parameter(torch.tensor(k2))

        # self.k1= (torch.tensor(k1))
        # self.k2= (torch.tensor(k2))

        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        self.threshold.requires_grad = True

        self.scale = (1 / (in_channels*out_channels))
        # signal_shape must be defined various with layers
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, signal_shape, dtype=torch.cfloat))

    def bi_mod_sigmoid(self, threshold_co, fft_co_len, sig_co):
        # threhold_func = 1/(1+torch.exp(-self.k1*(sig_co - threshold_co1*fft_co_len))+torch.exp(-self.k2*(sig_co - threshold_co2*fft_co_len)))
        threhold_func = 1/(1 + torch.exp(-0.5*(sig_co - threshold_co*fft_co_len)))
        return threhold_func

    def low_sigmoid_step(self, threshold_co,  fft_co):
      batch = fft_co.size(0)
      channel = fft_co.size(1)
      
      fft_co_len = fft_co.size(-1)
      sig_co = torch.arange(0, fft_co_len, 1).cuda()
      
      threhold_func = self.bi_mod_sigmoid(threshold_co, fft_co_len, sig_co)

      sig_co = (1 - threhold_func).to(torch.float)
      sig_co = sig_co.repeat(batch, channel, 1)

      fft_co = torch.mul(fft_co, sig_co)
      return fft_co

    def high_sigmoid_step(self, threshold_co, fft_co):
      batch = fft_co.size(0)
      channel = fft_co.size(1)

      fft_co_len = fft_co.size(-1)
      sig_co = torch.arange(0, fft_co_len, 1).cuda()
      
      threhold_func = self.bi_mod_sigmoid(threshold_co, fft_co_len, sig_co)

      sig_co = threhold_func.to(torch.float)
      sig_co = sig_co.repeat(batch, channel, 1)

      fft_co = torch.mul(fft_co, sig_co)
      return fft_co

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        
        x_ft = torch.fft.rfft(x)

        modes = x.size(-1)//2 + 1
        
        # clamp threshold
        self.threshold.data = self.threshold.clamp(min=0.001, max=1.)

        #Return to physical space
        x = torch.fft.irfft(self.low_sigmoid_step(self.threshold, x_ft[:, :, :modes]), n=x.size(-1))
        minus_x = torch.fft.irfft(self.high_sigmoid_step(self.threshold, x_ft[:, :, :modes]), n=x.size(-1))

        x = self.compl_mul1d(x[:, :, :], self.weights)
        minus_x = self.compl_mul1d(minus_x[:, :, :], self.weights)

        x, minus_x = F.hardswish(x), F.hardswish(minus_x)
        return x, minus_x

class ResNet_PTB(nn.Module):
    def __init__(self, ResBlock, SpectralConv1d, init_threshold, num_classes):
        super(ResNet_PTB, self).__init__()

        self.init_threhsold = init_threshold

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2) 

        # self.fft32 = SpectralConv1d(64, 64, init_threshold = 0.2, k1=self.k1, k2=self.k2) 
        # signal shape must be defined various in each fft layer
        # signal shape depends on the input length
        self.fft64 = SpectralConv1d(64, 64, init_threshold = 0.2, signal_shape = 498) 
        self.fft128 = SpectralConv1d(128, 128, init_threshold = 0.2, signal_shape = 256) 
        self.fft256 = SpectralConv1d(256, 256, init_threshold = 0.2, signal_shape = 128)
        self.fft512 = SpectralConv1d(512, 512, init_threshold= 0.2, signal_shape = 64) 
        
        # self.low_ratio, self.high_ratio = torch.tensor(-0.5) ,torch.tensor(0.5)
        self.low_ratio, self.high_ratio = nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))

        # self.DAT64 = ChannelAttention(64)
        # self.DAT128 = ChannelAttention(128)
        # self.DAT256 = ChannelAttention(256)
        # self.DAT512 = ChannelAttention(512)

        self.fc2 = nn.Linear(1024, num_classes)
        self.adapt_avg = nn.AdaptiveAvgPool1d(1)
        self.adapt_max = nn.AdaptiveMaxPool1d(1)

        
        # self.dropout = nn.Dropout(0.3)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):

        #-------------------------------------------------first block
        # x, _= self.att_int(x, x, x)
        out = self.conv1(x) 
        # low_fft, high_fft = self.fft32(out)
        # out = out + self.low_ratio*low_fft + self.high_ratio*high_fft

        #-------------------------------------------------Res block 1

        out = self.layer1(out)
        low_fft, high_fft = self.fft64(out)
        out = out + self.low_ratio*low_fft + self.high_ratio*high_fft
        # out = self.DAT64(out)

        #-------------------------------------------------Res block 2      
        
        out = self.layer2(out)
        low_fft, high_fft = self.fft128(out)
        out = out + self.low_ratio*low_fft + self.high_ratio*high_fft
        # out = self.DAT128(out)
 
        #-------------------------------------------------Res block 3
        
        out = self.layer3(out)
        low_fft, high_fft = self.fft256(out)
        out = out + self.low_ratio*low_fft + self.high_ratio*high_fft
        # out = self.DAT256(out)
 
        #-------------------------------------------------Res block 4
        
        out = self.layer4(out)
        low_fft, high_fft = self.fft512(out)
        out = out + self.low_ratio*low_fft + self.high_ratio*high_fft
        # out = self.DAT512(out)

        #-------------------------------------------------Pooling block
        out_1, out_2 = self.adapt_max(out), self.adapt_avg(out)
        out = torch.cat([out_1, out_2], dim=1)

        out = out.reshape(out.size(0), -1)

        #-------------------------------------------------FC block
        out = self.fc2(out)
        
        # out = self.dropout(out)

        return out