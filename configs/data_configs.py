from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['ffhq'],
		'test_target_root': dataset_paths['ffhq'],
		'unpaired_root': dataset_paths['ffhq'],
	},
	'sketch_to_celebs': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['sketch_train'],
		'train_target_root': dataset_paths['sketch_train'],
		'test_source_root': dataset_paths['sketch_test'],
		'test_target_root': dataset_paths['sketch_test'],
		'unpaired_root': dataset_paths['celeba_train'],
	},
	'sketch_to_ffhq': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['sketch_train'],
		'train_target_root': dataset_paths['sketch_train'],
		'test_source_root': dataset_paths['sketch_test'],
		'test_target_root': dataset_paths['sketch_test'],
		'unpaired_root': dataset_paths['ffhq'],
	},
	'celebs_seg_to_ffhq': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_segmentation'],
		'train_target_root': dataset_paths['celeba_train_segmentation'],
		'test_source_root': dataset_paths['celeba_test_segmentation'],
		'test_target_root': dataset_paths['celeba_test_segmentation'],
		'unpaired_root': dataset_paths['ffhq'],
	},
	'celebs_seg_to_celebs': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_segmentation'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test_segmentation'],
		'test_target_root': dataset_paths['celeba_test_segmentation'],
		'unpaired_root': dataset_paths['celeba_train'],
	},
	'afhqcat_to_afhqdog': {
		'transforms': transforms_config.RGBToRGBTransforms,
		'train_source_root': dataset_paths['afhq_cat_train'],
		'train_target_root': dataset_paths['afhq_cat_train'],
		'test_source_root': dataset_paths['afhq_cat_test'],
		'test_target_root': dataset_paths['afhq_cat_test'],
		'unpaired_root': dataset_paths['afhq_dog_train'],
	},
	'afhqdog_to_afhqcat': {
		'transforms': transforms_config.RGBToRGBTransforms,
		'train_source_root': dataset_paths['afhq_dog_train'],
		'train_target_root': dataset_paths['afhq_dog_train'],
		'test_source_root': dataset_paths['afhq_dog_test'],
		'test_target_root': dataset_paths['afhq_dog_test'],
		'unpaired_root': dataset_paths['afhq_cat_train'],
	},
}
