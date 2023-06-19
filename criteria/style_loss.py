import torch
import torch.nn as nn
import torch.nn.functional as F

from criteria.lpips.networks import get_network

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class StyleLoss(nn.Module):
    r"""from https://github.com/naoto0804/pytorch-AdaIN
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
    """
    def __init__(self, net_type: str = 'alex'):

        super(StyleLoss, self).__init__()
        self.net = get_network(net_type)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)
    
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return F.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return F.mse_loss(input_mean, target_mean) + \
               F.mse_loss(input_std, target_std)
    
    def calc_normalized_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        size = input.size()
        normalized_input = (input - input_mean.expand(
        size)) / input_std.expand(size)
        normalized_target = (target - target_mean.expand(
        size)) / target_std.expand(size)
    
        return F.mse_loss(normalized_input, normalized_target)
    
    

    
    
    
