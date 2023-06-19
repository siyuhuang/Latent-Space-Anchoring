"""
This file defines the core research contribution
"""
import os, sys
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from models.encoders import psp_encoders
from models.regressor import Regressor
from models.stylegan2.model import Generator
from configs.paths_config import model_paths



def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name and k[len(name)]=='.'}
    return d_filt


class GanToAny(nn.Module):
    def __init__(self, opts):
        super(GanToAny, self).__init__()
        self.set_opts(opts)
        self.z_dim = 512
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.output_size, 512, 8, channel_multiplier=self.opts.channel_multiplier)
        if self.opts.freezeG is not None:
            self.decoder2 = Generator(self.opts.output_size, 512, 8, channel_multiplier=self.opts.channel_multiplier)
        
        if self.opts.feat_aggregation=='single':
            self.regressor = Regressor(self.opts.channel_multiplier*128, self.opts.label_nc, network=self.opts.regressor)
        elif self.opts.feat_aggregation=='concat':
            self.regressor = Regressor(5888, self.opts.label_nc, network=self.opts.regressor)
        else:
            self.opts.feat_aggregation = int(self.opts.feat_aggregation)
            channels = [
            512,
            512,
            512,
            512,
            256 * channel_multiplier,
            128 * channel_multiplier,
            64 * channel_multiplier,
            32 * channel_multiplier,
            16 * channel_multiplier,
            ]
            self.regressor = Regressor(channels[self.opts.feat_aggregation]*2, self.opts.label_nc, network=self.opts.regressor)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        
        if not self.opts.feat_aggregation == 'single':
            res  = 256
            mode = 'bilinear'
            self.upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode),
                      nn.Upsample(scale_factor=res / 4, mode=mode),
                      nn.Upsample(scale_factor=res / 8, mode=mode),
                      nn.Upsample(scale_factor=res / 8, mode=mode),
                      nn.Upsample(scale_factor=res / 16, mode=mode),
                      nn.Upsample(scale_factor=res / 16, mode=mode),
                      nn.Upsample(scale_factor=res / 32, mode=mode),
                      nn.Upsample(scale_factor=res / 32, mode=mode),
                      nn.Upsample(scale_factor=res / 64, mode=mode),
                      nn.Upsample(scale_factor=res / 64, mode=mode),
                      nn.Upsample(scale_factor=res / 128, mode=mode),
                      nn.Upsample(scale_factor=res / 128, mode=mode),
                      nn.Upsample(scale_factor=res / 256, mode=mode),
                      nn.Upsample(scale_factor=res / 256, mode=mode),
                      ]
        
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.phase=='train':
            if self.opts.checkpoint_path is not None:
                print('Loading checkpoint: {}'.format(self.opts.checkpoint_path))
                ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
                ckpt = ckpt['translator']
                self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
                self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
                if self.opts.freezeG is not None:
                    self.decoder2.load_state_dict(get_keys(ckpt, 'decoder2'), strict=True)
                self.__load_latent_avg(ckpt)
            else:
                print('Loading encoders weights from irse50!')
                encoder_ckpt = torch.load(model_paths['ir_se50'])
                # if self.opts.input_nc != 3:
                # do not load the input layer weights
                encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
                self.encoder.load_state_dict(encoder_ckpt, strict=False)
                print('Loading pretrained decoder weights!')
                ckpt = torch.load(self.opts.stylegan_weights)
                self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
                if self.opts.freezeG is not None:
                    self.decoder2.load_state_dict(ckpt['g_ema'], strict=True)
                print('Loading pretrained average latent!')
                if self.opts.learn_in_w:
                    self.__load_latent_avg(ckpt, repeat=1)
                else:
                    self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)
        else:
            if os.path.isfile(self.opts.exp_input):
                encoder_path = self.opts.exp_input
            else:
                path = os.path.join(self.opts.exp_input,'checkpoint')
                if self.opts.ckpt_input is None:
                    encoder_path = os.path.join(path, 'best_model.pt')
                else:
                    encoder_path = os.path.join(path, 'iteration_'+str(self.opts.ckpt_input)+'.pt')
            encoder_ckpt = torch.load(encoder_path)
            self.__load_latent_avg(encoder_ckpt)
            print('Loaded pretrained average latent')
            
            if 'translator' in encoder_ckpt:
                encoder_ckpt = encoder_ckpt['translator']
            else:
                encoder_ckpt = encoder_ckpt['state_dict']
            self.encoder.load_state_dict(get_keys(encoder_ckpt, 'encoder'), strict=True)
            print('Loaded encoder weights from', encoder_path)
            
            if os.path.isfile(self.opts.exp_output):
                regressor_path = self.opts.exp_output
            else:
                path = os.path.join(self.opts.exp_output,'checkpoint')
                if self.opts.ckpt_output is None:
                    regressor_path = os.path.join(path, 'best_model.pt')
                else:
                    regressor_path = os.path.join(path, 'iteration_'+str(self.opts.ckpt_output)+'.pt')
            regressor_ckpt = torch.load(regressor_path)
            if 'translator' in regressor_ckpt:
                regressor_ckpt = regressor_ckpt['translator']
            else:
                regressor_ckpt = regressor_ckpt['state_dict']
            self.regressor.load_state_dict(get_keys(regressor_ckpt, 'regressor'), strict=True)
            print('Loaded regressor weights from', regressor_path)
            
            self.decoder.load_state_dict(get_keys(regressor_ckpt, 'decoder'), strict=True)
            
            if self.opts.freezeG is not None:
                print('Loaded decoder2 weights from', regressor_path)
                self.decoder2.load_state_dict(get_keys(regressor_ckpt, 'decoder2'), strict=True)
            print('Loaded StyleGAN generator weights')
           

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, alpha=None, just_return_codes=False):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
            
        if latent_mask is not None:
            if inject_latent is not None:
                for i in latent_mask:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
            else:
                noise = np.random.randn(codes.size(0)*18, 512).astype('float32')
                noise = torch.from_numpy(noise).to("cuda")
                styles = self.decoder.style(noise)
                styles = styles.reshape(codes.size(0), 18, -1)
                for i in latent_mask:
                    codes[:, i] = styles[:, i]

        if just_return_codes:
            return codes
                            
        input_is_latent = not input_code

        images, result_latent, result_features = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=True,
                                             return_features=True,
                                             layer=None)
        
        if self.opts.freezeG is not None:
            images2, _, _ = self.decoder2([codes],
                                         input_is_latent=input_is_latent,
                                         randomize_noise=randomize_noise,
                                         return_latents=True,
                                         return_features=True,
                                         layer=None)
    
        if self.opts.feat_aggregation=='single':
            regressor_input = torch.cat((result_features[12], result_features[13]),1)
        elif self.opts.feat_aggregation=='concat':
            regressor_input = self.upsamplers[0](result_features[0])
            for i in range(1,14):
                regressor_input = torch.cat((regressor_input, self.upsamplers[i](result_features[i])),1)
        else: 
            regressor_input = torch.cat(
                (self.upsamplers[self.opts.feat_aggregation*2](result_features[self.opts.feat_aggregation*2]),
                self.upsamplers[self.opts.feat_aggregation*2+1](result_features[self.opts.feat_aggregation*2+1])),
                1)
        if 'attention' in self.opts.regressor:
            seg, attention_weights = self.regressor(regressor_input)
        else:
            seg = self.regressor(regressor_input)
        
        if resize:
            resized_images = self.face_pool(images)
            
        results = {'x_hat': seg, 'y_hat': resized_images, 
                'latent': result_latent, 'y_hat_highres': images,
                'features': result_features}
        if 'attention' in self.opts.regressor:
            results['attention_weights'] = attention_weights
            
        if self.opts.freezeG is not None:
            results['x_hat'] = resized_images
            resized_images2 = self.face_pool(images2)
            results['y_hat'] = resized_images2
            results['y_hat_highres'] = images2
            
        return results
    
    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None