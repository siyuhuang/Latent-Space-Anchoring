# segmentation_to_ffhq
python scripts/test.py \
--phase=inference \
--dataset_type=celebs_seg_to_ffhq \
--exp_input=./pretrained_models/seg2ffhq.pt \
--exp_output=./pretrained_models/seg2ffhq.pt \
--result_dir=./results/seg2ffhq_inference \
--regressor=deeper \
--input_nc=19 \
--label_nc=19 \
--workers=1 \
--batch_size=1 \
--test_batch_size=1 \
--test_workers=1 \
--network_type=gan_to_any \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg 
