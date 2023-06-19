# ffhq_to_segmentation
python scripts/test.py \
--phase=test \
--dataset_type=ffhq_encode \
--exp_input=./pretrained_models/psp_ffhq_encode.pt \
--exp_output=./pretrained_models/sketch2ffhq.pt \
--result_dir=./results/ffhq2sketch \
--regressor=deeper \
--input_nc=3 \
--label_nc=1 \
--workers=1 \
--batch_size=1 \
--test_batch_size=4 \
--test_workers=1 \
--network_type=gan_to_any \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg 