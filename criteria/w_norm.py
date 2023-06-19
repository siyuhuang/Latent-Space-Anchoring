import torch
from torch import nn


class WNormLoss(nn.Module):
    def __init__(self, start_from_latent_avg=True):
        super(WNormLoss, self).__init__()
        self.start_from_latent_avg = start_from_latent_avg

    def forward(self, latent, latent_avg=None):
        if self.start_from_latent_avg:
            latent = latent - latent_avg
        return torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]
    
class WVarLoss(nn.Module):
    def __init__(self):
        super(WVarLoss, self).__init__()

    def forward(self, latent, latent_avg, latent_std):
        loss_mean = torch.sum(latent.norm(2, dim=(1, 2))) / latent.shape[0]
        loss_std = (latent.std([0,1], unbiased=True)-latent_std).norm(1)/512.
        return loss_mean, loss_std
    
        if self.latent_avg is not None:
            out = input - self.latent_avg.repeat(input.shape[0], 1)

            group = min(batch, self.stddev_group)
            stddev = out.view(
                group, -1, self.stddev_feat, channel // self.stddev_feat
            )
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean(2, keepdims=False)
            stddev = stddev.repeat(group, 1)
