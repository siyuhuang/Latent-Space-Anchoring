dataset_paths = {
	'celeba_train': 'data/CelebAMask-HQ/face_parsing/Data_preprocessing/train_img',
	'celeba_test': 'data/CelebAMask-HQ/face_parsing/Data_preprocessing/test_img',
	'celeba_train_segmentation': 'data/CelebAMask-HQ/face_parsing/Data_preprocessing/train_label',
	'celeba_test_segmentation': 'data/CelebAMask-HQ/face_parsing/Data_preprocessing/test_label',
	'sketch_train': 'data/CUFSF/train',
	'sketch_test': 'data/CUFSF/test',
	'ffhq': 'data/ffhq/images1024x1024',

	'afhq_cat_train': 'data/AFHQ/afhq/train/cat',
	'afhq_cat_test': 'data/AFHQ/afhq/val/cat',
	'afhq_dog_train': 'data/AFHQ/afhq/train/dog',
	'afhq_dog_test': 'data/AFHQ/afhq/val/dog',
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'stylegan_celebahq': 'pretrained_models/stylegan2-celeba-hq-256x256.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
}
