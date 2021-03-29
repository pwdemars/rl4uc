#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np

from rl4uc.environment import make_env

def test_make_train_env():
	for g in [5,10,20,30]:
		env = make_env(mode='train', num_gen=g)
		obs = env.reset()

def test_make_test_env():
	df = pd.DataFrame({'demand': np.array([600., 550., 650.,]),
				  	   'wind': np.array([10., 50., 70.])})
	env = make_env(mode='test', profiles_df = df)
	obs = env.reset()