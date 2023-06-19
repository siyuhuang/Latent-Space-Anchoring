# segmentation_to_sketch
python scripts/test.py \
--phase=test \
--dataset_type=celebs_seg_to_ffhq \
--exp_input=./pretrained_models/seg2ffhq.pt \
--exp_output=./pretrained_models/sketch2ffhq.pt \
--result_dir=./results/seg2sketch \
--regressor=deeper \
--input_nc=19 \
--label_nc=1 \
--workers=1 \
--batch_size=1 \
--test_batch_size=4 \
--test_workers=1 \
--network_type=gan_to_any \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg 
