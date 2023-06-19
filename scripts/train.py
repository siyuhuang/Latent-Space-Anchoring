"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions



def main():
	opts = TrainOptions().parse()
	if not os.path.exists(opts.exp_dir):
		os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	from training.coach_bimodal import Coach 
	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
