"""
This file runs the inference loop
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
	opts_dict = vars(opts)
	if opts.network_type=='psp': 
		from training.coach import Coach
	else:
		from training.coach_bimodal import Coach 
	coach = Coach(opts)
	coach.sampling()


if __name__ == '__main__':
	main()
