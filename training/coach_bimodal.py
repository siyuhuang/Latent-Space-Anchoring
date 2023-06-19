import os, sys
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.use('Agg')

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from utils.train_utils import requires_grad, requires_grad_target_layer
from criteria import id_loss, w_norm, moco_loss
from criteria.d_loss import d_logistic_loss
from configs import data_configs
from criteria.style_loss import StyleLoss
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from models.stylegan2.model import Discriminator, LinearDiscriminator
from training.ranger import Ranger


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device
        
        if self.opts.use_wandb:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)

        # Initialize network
        if self.opts.network_type=='gan_to_any':
            from models.gan_to_any import GanToAny
        else:
            raise Exception('{} is not a valid network'.format(self.opts.network_type))
        self.net = GanToAny(self.opts).to(self.device)
        
        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            if self.opts.start_from_latent_avg:
                self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()
            else:
                self.net.latent_avg = None
            
        if opts.phase=='train' and opts.imageD_lambda > 0:
            self.imageD = Discriminator(256, channel_multiplier=self.opts.channel_multiplier).to(self.device)
            if self.opts.checkpoint_path is not None:
                from models import gan_to_any
                ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
                self.imageD.load_state_dict(ckpt['imageD'], strict=True)
        if opts.phase=='train' and opts.latentD_lambda > 0:
            self.latentD = LinearDiscriminator(input_dim=512, latent_avg=self.net.latent_avg).to(self.device)
        
        # Initialize loss
        if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
            raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
        if self.opts.moco_lambda > 0:
            self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()
        if self.opts.seg_lambda > 0:
            self.seg_loss = nn.CrossEntropyLoss().to(self.device).eval()
        if self.opts.imageC_lambda + self.opts.reconC_lambda + self.opts.reconS_lambda > 0:
            self.style_loss = StyleLoss(net_type='vgg').to(self.device).eval()

        # Initialize optimizer
        if opts.phase=='train':
            self.optimizer = self.configure_optimizers()
            if self.opts.latentD_lambda > 0:
                self.optimizer_latentD = Ranger(list(self.latentD.parameters()), lr=self.opts.learning_rate)
            if self.opts.imageD_lambda > 0:
                self.optimizer_imageD = Ranger(list(self.imageD.parameters()), lr=self.opts.learning_rate)

        # Initialize dataset
        self.train_dataset, self.test_dataset, self.unpaired_dataset = self.configure_datasets()
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True,
                                           pin_memory=True)
        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)
        if self.unpaired_dataset is not None:
            self.unpaired_dataloader = DataLoader(self.unpaired_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True,
                                           pin_memory=True)

        # Initialize logger
        if opts.phase=='train':
            log_dir = os.path.join(opts.exp_dir)
            os.makedirs(log_dir, exist_ok=True)
            self.logger = SummaryWriter(log_dir=log_dir)
            self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoint')
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.best_val_loss = None
            if self.opts.save_interval is None:
                self.opts.save_interval = self.opts.max_steps
        if opts.phase in ['inference', 'visualize', 'test', 'validate', 'sampling']:
            log_dir = opts.result_dir
            os.makedirs(log_dir, exist_ok=True)
            self.logger = SummaryWriter(log_dir=log_dir)

    def train(self):
        self.net.train()
        
        train_dataloader_iterator = iter(self.train_dataloader)
        if hasattr(self, 'unpaired_dataloader'):
            unpaired_dataloader_iterator = iter(self.unpaired_dataloader)
        while self.global_step < self.opts.max_steps:
            loss_dict = {}
            try:
                x = next(train_dataloader_iterator)
            except:
                train_dataloader_iterator = iter(self.train_dataloader)
                x = next(train_dataloader_iterator)

            if type(x) is list:
                x, label = x[0], x[1]
                label = label.view(x.size(0),-1).to(self.device)
            else:
                label = None
            x = x.to(self.device).float()
            if label is not None:
                r = self.net.forward(x, label)
            else:
                r = self.net.forward(x)
            
            # train latent discriminator
            if self.opts.latentD_lambda > 0: 
                self.optimizer_latentD.zero_grad()
                requires_grad(self.net, False)
                requires_grad(self.latentD, True)

                fake_latent = self.net.forward(x.detach(), just_return_codes=True).view(-1, 512)
                noise = torch.randn(fake_latent.size(0), 512, device=self.device)
                real_latent = self.net.decoder.style(noise.detach())

                real_pred = self.latentD(real_latent.detach())
                fake_pred = self.latentD(fake_latent.detach())
                loss_latentD_real, loss_latentD_fake = d_logistic_loss(real_pred, fake_pred)
                loss_dict["loss_latentD_real"] = round(float(loss_latentD_real), 3)
                loss_dict["loss_latentD_fake"] = round(float(loss_latentD_fake), 3)

                loss_latentD = (loss_latentD_real + loss_latentD_fake) * self.opts.latentD_lambda
                loss_latentD.backward()
                self.optimizer_latentD.step()
            
            # train image discriminator
            if self.opts.imageD_lambda > 0: 
                try:
                    x_unpaired = next(unpaired_dataloader_iterator)
                except:
                    unpaired_dataloader_iterator = iter(self.unpaired_dataloader)
                    x_unpaired = next(unpaired_dataloader_iterator)
                    
                if type(x_unpaired) is list:
                    x_unpaired, label_unpaired = x_unpaired[0], x_unpaired[1]
                    label_unpaired = label_unpaired.view(x_unpaired.size(0),-1).to(self.device)
                else:
                    label_unpaired = None
                x_unpaired = x_unpaired.to(self.device).float()
                
                self.optimizer_imageD.zero_grad()
                requires_grad(self.net, False)
                requires_grad(self.imageD, True)
                
                real_pred = self.imageD(x_unpaired.detach())
                fake_pred = self.imageD(r['y_hat'].detach())
                loss_imageD_real, loss_imageD_fake = d_logistic_loss(real_pred, fake_pred)
                loss_dict["loss_imageD_real"] = round(float(loss_imageD_real), 3)
                loss_dict["loss_imageD_fake"] = round(float(loss_imageD_fake), 3)
                
                loss_imageD = (loss_imageD_real + loss_imageD_fake) * self.opts.imageD_lambda
                loss_imageD.backward()
                self.optimizer_imageD.step()
          
            self.optimizer.zero_grad()
            requires_grad(self.net, True)
            requires_grad(self.net.decoder, False)
            if self.opts.freezeG is not None:
                requires_grad(self.net.decoder2, False)
                for layer in range(self.opts.freezeG):
                    requires_grad_target_layer(self.net.decoder, True, target_layer=f'convs.{self.net.decoder.num_layers-2-2*layer}')
                    requires_grad_target_layer(self.net.decoder, True, target_layer=f'convs.{self.net.decoder.num_layers-3-2*layer}')
                    requires_grad_target_layer(self.net.decoder, True, target_layer=f'to_rgbs.{self.net.decoder.log_size-3-layer}') 
            if hasattr(self, 'latentD'):
                requires_grad(self.latentD, False)
            if hasattr(self, 'imageD'):
                requires_grad(self.imageD, False)
            
            # train G  
            loss, loss_dict, id_logs = self.calc_loss(x, r, loss_dict=loss_dict)
            loss.backward()
            self.optimizer.step()

            # Logging related
            if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                self.parse_and_log_images(id_logs, 
                                  x, r['x_hat'], r['y_hat'], 
                                  title='train', 
                                  display_count=x.size(0)
                                 )
            if self.global_step % self.opts.board_interval == 0:
                self.print_metrics(loss_dict, prefix='train')
                self.log_metrics(loss_dict, prefix='train')
                
            # Validation related
            val_loss_dict = None
            if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                val_loss_dict = self.validate()
                if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                    self.best_val_loss = val_loss_dict['loss']
                    self.checkpoint_me(val_loss_dict, is_best=True)

            if self.global_step > 0 and self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                if val_loss_dict is not None:
                    self.checkpoint_me(val_loss_dict, is_best=False)
                else:
                    self.checkpoint_me(loss_dict, is_best=False)

            if self.global_step == self.opts.max_steps:
                print('OMG, finished training!')
                break

            self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                x = batch
                if type(x) is list:
                    x, label = x[0], x[1]
                    label = label.view(x.size(0),-1).to(self.device)
                else:
                    label = None
                x = x.to(self.device).float()
                if label is not None:
                    r = self.net.forward(x, label)
                else:
                    r = self.net.forward(x)
                loss, cur_loss_dict, id_logs = self.calc_loss(x, r)
                
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(id_logs, x, r['x_hat'], r['y_hat'],
                                      title='test',
                                      subscript='{:04d}'.format(batch_idx),
                                      display_count=x.size(0)
                                       )

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 2:
                self.net.train()
                return None  # Do not log, inaccurate in first batch
            
            if batch_idx >= 20:
                break
            
        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict
    
    def inference(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x = batch
            with torch.no_grad():
                if type(x) is list:
                    x, label = x[0], x[1]
                    label = label.view(x.size(0),-1).to(self.device)
                else:
                    label = None
                x = x.to(self.device).float()
                if label is not None:
                    r = self.net.forward(x, label)
                else:
                    r = self.net.forward(x)
                
                loss, cur_loss_dict, id_logs = self.calc_loss(x, r)
                
                mixed_outputs = []
                if label is None: # only mixing styles on StyleGAN
                    for i in range(5):
                        mixed_output = self.net.forward(x, 
                                           latent_mask=[6,7,8,9,10,11,12,13,14,15,16,17])
                        mixed_outputs.append(mixed_output)
                
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images_inference(x, r['x_hat'], r['y_hat'], mixed_outputs,
                                              title='images',
                                              subscript='{:04d}'.format(batch_idx)
                                               )
            
            if batch_idx >= 20:
                break

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        return loss_dict
    
    def test(self):
        self.net.eval()
        for batch_idx, batch in enumerate(self.test_dataloader):
            x = batch
            with torch.no_grad():
                if type(x) is list:
                    x, label = x[0], x[1]
                    label = label.view(x.size(0),-1).to(self.device)
                else:
                    label = None
                x = x.to(self.device).float()
                if label is not None:
                    r = self.net.forward(x, label)
                else:
                    r = self.net.forward(x)

            # Logging related
            save_file = self.test_dataset.source_paths[batch_idx]
            if isinstance(save_file, list):
                save_file = save_file[0]
            save_file = os.path.basename(save_file)
            save_file = os.path.splitext(save_file)[0]
            self.parse_and_log_images_test(x, r['x_hat'], r['y_hat'], 
                                              title='images',
                                              subscript='{:04d}'.format(batch_idx)
                                               )
            if batch_idx >= 20:
                break
    
    def sampling(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                
                x = torch.randn(batch.size(0), 512).to(self.device).float()*0.01
                x = self.net.decoder.style(x)
                x = x.unsqueeze(1).repeat(1, 18, 1)
                r = self.net.forward(x, input_code=True)
                
                # get mixing styles
                mixed_outputs = []
                for i in range(5):
                    mixed_output = self.net.forward(x.detach(), 
                                       input_code=True,
                                       latent_mask=[12,13,14,15,16,17])
                    mixed_outputs.append(mixed_output)

            # Logging related
            self.parse_and_log_images_sampling(r, mixed_outputs,
                                              title='images',
                                              subscript='{:04d}'.format(batch_idx)
                                               )
            del x, r, mixed_outputs
            
            if batch_idx >= 20:
                break

    
    def visualize(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x = batch
            with torch.no_grad():
                x = x.to(self.device).float()
                r = self.net.forward(x)
                
            # Logging related
            self.parse_and_log_images_visualize(x, r['x_hat'], r['y_hat'], 
                                                r,
                                              title='images',
                                              subscript='{:04d}'.format(batch_idx)
                                               )
            
            if batch_idx >= 20:
                break

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.opts.exp_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if hasattr(self.net, 'regressor'):
            params += list(self.net.regressor.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())

        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        print(f'Loading datasets for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        if 'imagenet' in self.opts.dataset_type:
            from datasets.images_dataset import ImagenetDataset as Dataset
        elif 'landmark' in self.opts.dataset_type:
            from datasets.images_dataset import LandmarksDataset as Dataset
        else:
            from datasets.images_dataset import ImagesDataset as Dataset 
            
        if self.opts.sr_ratio is not None:
            import torchvision.transforms as transforms
            transforms_dict['transform_source'] = transforms.Compose([
                                transforms.Resize((256//self.opts.sr_ratio, 256//self.opts.sr_ratio)),
                                transforms.Resize((256, 256)),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        if self.opts.phase == 'train':
            train_dataset = Dataset(source_root=dataset_args['train_source_root'],
                                      source_transform=transforms_dict['transform_source'],
                                      opts=self.opts,
                                      preload=self.opts.preload)
        else:
            train_dataset = None
            
        test_dataset = Dataset(source_root=dataset_args['test_source_root'],
                                     source_transform=transforms_dict['transform_test'],
                                     opts=self.opts,
                                     phase='test')
        
        if self.opts.phase == 'train' and self.opts.imageD_lambda > 0:
            if 'imagenet' in self.opts.dataset_type:
                from datasets.images_dataset import ImagenetDataset as UnpairedImagesDataset
            else:
                from datasets.images_dataset import UnpairedImagesDataset
            unpaired_dataset = UnpairedImagesDataset(source_root=dataset_args['unpaired_root'],
                                         source_transform=transforms_dict['transform_unpaired'],
                                         opts=self.opts,
                                         input_nc=3)
            print(f"Number of unpaired samples: {len(unpaired_dataset)}")
        else:
            unpaired_dataset = None
            
            
        if self.opts.use_wandb:
            if train_dataset is not None:
                self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
            self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
        if train_dataset is not None:
            print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset, unpaired_dataset

    def calc_loss(self, x, r, loss_dict={}):
        x_hat, y_hat, latent = r['x_hat'], r['y_hat'], r['latent']
        loss = 0.0
        id_logs = None
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(x_hat, x)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(x_hat, x)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.w_norm_lambda > 0:
            if self.net.latent_avg is not None:
                loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
            else:
                loss_w_norm = torch.sum(latent.norm(2, dim=1))/latent.size(0)
            loss_dict['loss_w_norm'] = round(float(loss_w_norm), 3)
            loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.attention_norm_lambda > 0 and 'attention' in self.opts.regressor:
            loss_attention_norm = torch.sum(r['attention_weights'].norm(1, dim=1))/latent.size(0)
            loss_dict['loss_attention_norm'] = round(float(loss_attention_norm), 3)
            loss += loss_attention_norm * self.opts.attention_norm_lambda
        if self.opts.seg_lambda > 0:
            target = x.argmax(dim=1)
            loss_seg = self.seg_loss(x_hat, target)
            loss_dict['loss_seg'] = round(float(loss_seg), 3)
            loss += loss_seg * self.opts.seg_lambda
        if self.opts.sketch_lambda > 0:
            loss_sketch = F.mse_loss(x_hat, x)
            loss_dict['loss_sketch'] = round(float(loss_sketch), 3)
            loss += loss_sketch * self.opts.sketch_lambda
        if self.opts.landmark_lambda > 0:
            loss_landmark = F.mse_loss(x_hat, x)
            loss_dict['loss_landmark'] = round(float(loss_landmark), 3)
            loss += loss_landmark * self.opts.landmark_lambda
        if self.opts.phase == 'train' and self.opts.imageD_lambda > 0:
            imageD_pred = self.imageD(r['y_hat'])
            loss_imageD_G = F.softplus(-imageD_pred).mean()
            loss_dict['loss_imageD_G'] = round(float(loss_imageD_G), 3)
            loss += loss_imageD_G * self.opts.imageD_lambda
        if self.opts.phase == 'train' and self.opts.latentD_lambda > 0:
            latentD_pred = self.latentD(latent.view(-1, 512))
            loss_latentD_G = F.softplus(-latentD_pred).mean()
            loss_dict['loss_latentD_G'] = round(float(loss_latentD_G), 3)
            loss += loss_latentD_G * self.opts.latentD_lambda
        if self.opts.imageC_lambda + self.opts.reconC_lambda + self.opts.reconS_lambda > 0:
            feat_x = self.style_loss.net(x)
            if self.opts.imageC_lambda > 0:
                feat_y_hat = self.style_loss.net(y_hat)
                loss_imageC = self.style_loss.calc_normalized_content_loss(feat_y_hat[-2], feat_x[-2])
                loss_dict['loss_imageC'] = round(float(loss_imageC), 3)
                loss += loss_imageC * self.opts.imageC_lambda
            if self.opts.reconC_lambda + self.opts.reconS_lambda > 0:
                feat_x_hat = self.style_loss.net(x_hat)
                if self.opts.reconC_lambda > 0:
                    loss_reconC = self.style_loss.calc_content_loss(feat_x_hat[-2], feat_x[-2])
                    loss_dict['loss_reconC'] = round(float(loss_reconC), 3)
                    loss += loss_reconC * self.opts.reconC_lambda
                if self.opts.reconS_lambda > 0:
                    loss_reconS = self.style_loss.calc_style_loss(feat_x_hat[0], feat_x[0])
                    for i in range(1, 4):
                        loss_reconS = loss_reconS + self.style_loss.calc_style_loss(feat_x_hat[i], feat_x[i])
                    loss_dict['loss_reconS'] = round(float(loss_reconS), 3)
                    loss += loss_reconS * self.opts.reconS_lambda
                    
                    
        loss_dict['loss'] = round(float(loss), 3)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, x_hat, y_hat, title, subscript=None, display_count=4):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input': common.log_input_image(x[i], nc=self.opts.input_nc),
                'output_face': common.tensor2im(y_hat[i]),
                'reconstruct': common.log_input_image(x_hat[i], nc=self.opts.label_nc),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)
        
    def parse_and_log_images_inference(self, x, x_hat, y_hat, mixed_outputs, title, subscript=None):
        im = Image.new('RGB', ((3+len(mixed_outputs)*2)*256, x.size(0)*256))
        for i in range(x.size(0)):
            input = common.log_input_image(x[i], nc=self.opts.input_nc)
            im.paste(input, (0, i*256))
            reconstruct = common.log_input_image(x_hat[i], nc=self.opts.label_nc)
            im.paste(reconstruct, (1*256, i*256))
            face = common.tensor2im(y_hat[i])
            im.paste(face, (2*256, i*256))                        
            for j in range(len(mixed_outputs)):
                face = common.log_input_image(mixed_outputs[j]['x_hat'][i], nc=self.opts.label_nc)
                im.paste(face, ((j*2+3)*256,i*256))
            for j in range(len(mixed_outputs)):
                face = common.tensor2im(mixed_outputs[j]['y_hat'][i])
                im.paste(face, ((j*2+1+3)*256,i*256))
                    
        step = self.global_step
        path = os.path.join(self.logger.log_dir, 'images', f'{subscript}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        im.save(path)
        
    def parse_and_log_images_test(self, x, x_hat, y_hat, title, subscript=None):
        im = Image.new('RGB', (3*256, x.size(0)*256))
        for i in range(x.size(0)):
            input = common.log_input_image(x[i], nc=self.opts.input_nc)
            im.paste(input, (0, i*256))
            reconstruct = common.log_input_image(x_hat[i], nc=self.opts.label_nc)
            im.paste(reconstruct, (1*256, i*256))
            face = common.tensor2im(y_hat[i])
            im.paste(face, (2*256, i*256))    
                    
        step = self.global_step
        path = os.path.join(self.logger.log_dir, 'images', f'{subscript}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        im.save(path)
        
    def parse_and_log_images_sampling(self, r, mixed_outputs, title, subscript=None):
        im = Image.new('RGB', ((1+len(mixed_outputs))*1024, r['x_hat'].size(0)*1024))
        for i in range(r['x_hat'].size(0)):
            input = common.log_input_image(r['x_hat'][i], nc=19, size=1024)
            im.paste(input, (0, i*1024))
            input = common.log_input_image(r['y_hat_highres'][i], nc=3, size=1024)
            im.paste(input, (1*1024, i*1024))
                      
            for j in range(len(mixed_outputs)):
                face = common.log_input_image(mixed_outputs[j]['y_hat_highres'][i], nc=3, size=1024)
                im.paste(face, ((j+2)*1024,i*1024))
                    
        step = self.global_step
        path = os.path.join(self.logger.log_dir, 'images', f'{subscript}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        im.save(path)
        
    def parse_and_log_images_visualize(self, x, x_hat, y_hat, r, title, subscript=None):
        im = Image.new('RGB', ((3+20)*256, x.size(0)*256))
        if 'attention_weights' in r.keys():
            attention_weights = r['attention_weights'].squeeze(-1).squeeze(-1).detach().cpu().numpy()
            weight_index = np.argsort(attention_weights, axis=1)
        for i in range(x.size(0)):
            input = common.log_input_image(x[i], nc=self.opts.input_nc)
            im.paste(input, (0, i*256))
            reconstruct = common.log_input_image(x_hat[i], nc=self.opts.label_nc)
            im.paste(reconstruct, (1*256, i*256))
            face = common.tensor2im(y_hat[i])
            im.paste(face, (2*256, i*256))
            features = torch.cat((r['features'][12], r['features'][13]), 1)
            if 'attention_weights' in r.keys():
                for j in range(10):
                    face = common.log_input_image(features[i,weight_index[i,-j-1]], nc='featmap')
                    im.paste(face, ((j+3)*256,i*256))
                for j in range(10):
                    face = common.log_input_image(features[i,weight_index[i,j]], nc='featmap')
                    im.paste(face, ((j+10+3)*256,i*256))
            else:
                for j in range(20):
                    face = common.log_input_image(features[i,j], nc='featmap')
                    im.paste(face, ((j+3)*256,i*256))
                    
        step = self.global_step
        path = os.path.join(self.logger.log_dir, 'images', f'{subscript}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        im.save(path)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data, self.opts.dataset_type)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'translator': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        
        if hasattr(self, 'latentD'):
            save_dict['latentD'] = self.latentD.state_dict()
        if hasattr(self, 'imageD'):
            save_dict['imageD'] = self.imageD.state_dict()
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict
