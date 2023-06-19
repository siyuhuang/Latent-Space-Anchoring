
import torch
import torch.nn as nn

from models.stylegan2.model import ResBlock


class Regressor(nn.Module):
    def __init__(self, input_dim, out_class, network='shallow'):
        super(Regressor, self).__init__()
        if network=='shallow':
            if out_class < 32:
                self.layers = nn.Sequential(
                    nn.Conv2d(input_dim, 128, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=128),
                    nn.Conv2d(128, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, out_class, 1),
                    # nn.Sigmoid()
                )
            else:
                self.layers = nn.Sequential(
                    nn.Conv2d(input_dim, 256, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=256),
                    nn.Conv2d(256, 128, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=128),
                    nn.Conv2d(128, out_class, 1),
                )
        elif network=='deep':
            self.layers = nn.Sequential(
                    nn.Conv2d(input_dim, 128, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=128),
                    nn.Conv2d(128, 64, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=64),
                    nn.Conv2d(64, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, out_class, 1),
                    # nn.Sigmoid()
                )
        elif network=='deeper':
            self.layers = nn.Sequential(
                    nn.Conv2d(input_dim, 128, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=128),
                    nn.Conv2d(128, 64, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=64),
                    nn.Conv2d(64, 64, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=64),
                    nn.Conv2d(64, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, out_class, 1),
                    # nn.Sigmoid()
                )
        elif network=='deep14':
            self.layers = nn.Sequential(
                    nn.Conv2d(input_dim, 128, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=128),
                    nn.Conv2d(128, 128, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=128),
                    nn.Conv2d(128, 128, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=128),
                    nn.Conv2d(128, 128, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=128),
                    nn.Conv2d(128, 64, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=64),
                    nn.Conv2d(64, 64, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=64),
                    nn.Conv2d(64, 64, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=64),
                    nn.Conv2d(64, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, out_class, 1),
                    # nn.Sigmoid()
                )
        elif network=='residual':
            self.layers = nn.Sequential(
                    ResBlock(input_dim, 128, downsample=False),
                    ResBlock(128, 64, downsample=False),
                    ResBlock(64, 64, downsample=False),
                    ResBlock(64, 32, downsample=False),
                    ResBlock(32, 32, downsample=False),
                    ResBlock(32, 32, downsample=False),
                    nn.Conv2d(32, out_class, 1),
                )
        elif network=='attention':
            self.layers = nn.Sequential(
                    nn.Conv2d(input_dim, 128, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=128),
                    nn.Conv2d(128, 64, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=64),
                    nn.Conv2d(64, 64, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=64),
                    nn.Conv2d(64, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, 32, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=32),
                    nn.Conv2d(32, out_class, 1),
                    # nn.Sigmoid()
                )
            self.CA = ChannelAttention(input_dim)

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)



    def forward(self, x):
        if hasattr(self, 'CA'):
            x, w = self.CA(x)
            return self.layers(x), w
        else:
            return self.layers(x)
    
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x, return_weights=True):
        y = self.conv_du(x)
        if return_weights:
            return x * y, y
        else:
            return x * y
        
        
        