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
	from training.coach_bimodal import Coach 
	coach = Coach(opts)
	if opts.phase == 'validate':
		coach.validate()
	elif opts.phase == 'inference':
		coach.inference()
	elif opts.phase == 'test':
		coach.test()
	elif opts.phase == 'sampling':
		coach.sampling()
	elif opts.phase == 'visualize':
		coach.visualize()

if __name__ == '__main__':
	main()
