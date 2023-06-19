from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--phase', default='train', type=str, choices=['train', 'test', 'inference', 'visualize', 'progressive', 'validate', 'sampling'], help='phase')
		self.parser.add_argument('--exp_input', default=None, type=str, help='experiment dir of input modality')
		self.parser.add_argument('--exp_output', default=None, type=str, help='experiment dir of output modality')
		self.parser.add_argument('--exp_dir', default=None, type=str, help='Path to experiment output directory')
		self.parser.add_argument('--ckpt_input', default=None, type=int, help='specify the loaded input checkpoint iteration')
		self.parser.add_argument('--ckpt_output', default=None, type=int, help='specify the loaded output checkpoint iteration')
		self.parser.add_argument('--result_dir', default='./results/', type=str, help='Path to inference results directory')
		self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--network_type', type=str, help='Which framework to use')
		self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
		self.parser.add_argument('--imagenet_class', default=None, type=str, help='Specify ImageNet class')
		self.parser.add_argument('--face_attribute', default=None, type=str, help='Specify CelebA face attribute class')
		self.parser.add_argument('--face_attribute_value', default=1, type=int, help='Specify CelebA face attribute value in {1,-1}')
		self.parser.add_argument('--sr_ratio', default=None, type=int, help='Ratio for Superresolution Task')
        
		self.parser.add_argument('--preload', action='store_true', help='Whether to preload training and test datasets into memory.')
		self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
		self.parser.add_argument('--input_inversion', action='store_true', help='Whether to invert input')

		self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
		self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
		self.parser.add_argument('--feat_aggregation', default='single', type=str, help='aggregation method of StyleGAN features')
		self.parser.add_argument('--regressor', default='shallow', type=str, help='arch of regressor')

		self.parser.add_argument('--lpips_lambda', default=0.0, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0.0, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=0.0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--w_norm_lambda', default=0.005, type=float, help='W-norm loss multiplier factor')
		self.parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
		self.parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')
		self.parser.add_argument('--moco_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--seg_lambda', default=0, type=float, help='segmentation reconstruction loss multiplier factor')
		self.parser.add_argument('--sketch_lambda', default=0, type=float, help='sketch reconstruction loss multiplier factor')
		self.parser.add_argument('--landmark_lambda', default=0, type=float, help='landmark reconstruction loss multiplier factor')
		self.parser.add_argument('--imageD_lambda', default=0.0, type=float, help='image discriminator loss multiplier factor')
		self.parser.add_argument('--latentD_lambda', default=0.0, type=float, help='latent discriminator loss multiplier factor')
		self.parser.add_argument('--imageC_lambda', default=0.0, type=float, help='image content loss multiplier factor')
		self.parser.add_argument('--reconC_lambda', default=0.0, type=float, help='reconstruction content loss multiplier factor')
		self.parser.add_argument('--reconS_lambda', default=0.0, type=float, help='reconstruction style loss multiplier factor')
		self.parser.add_argument('--attention_norm_lambda', default=0.0, type=float, help='attention weights norm multiplier factor')   
		self.parser.add_argument('--freezeG', default=None, type=int, help='unfreeze generator layers from this param')
		self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to model checkpoint')
		self.parser.add_argument('--channel_multiplier', default=2, type=int, help='Channel multiplier of StyleGAN2')

		self.parser.add_argument('--max_steps', default=300000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

		# arguments for weights & biases support
		self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None, help='For super-res, comma-separated resize factors to use for inference.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts
