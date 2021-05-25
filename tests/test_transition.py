#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from rl4uc.environment import make_env

@pytest.fixture
def example_test_profile():
    df = pd.DataFrame({'demand': np.array([600., 550., 650.,]),
				  	   'wind': np.array([10., 50., 70.])})
    return df

@pytest.fixture
def example_test_env(example_test_profile):
	return make_env(mode='test', profiles_df=example_test_profile, num_gen=5)

def test_reward_stochastic():
	np.random.seed(1)
	env = make_env(num_gen=5)
	env.reset()

	obs, reward, done = env.step(np.array([1,0,0,1,1]))
	assert reward == -9457.870355350562, "reward was: {}".format(reward)

def test_status_update():
	np.random.seed(1)
	env = make_env(num_gen=5)
	env.reset()
	obs, reward, done = env.step(np.ones(env.num_gen))
	assert np.all(env.status == np.array([1, 11, 13,  1,  3]))

def test_reward_deterministic(example_test_env):
	example_test_env.reset()
	obs, reward, done = example_test_env.step(np.ones(example_test_env.num_gen), deterministic=True)
	assert reward == -8471.23662554931, "reward was: {}".format(reward)

def test_terminal(example_test_env):
	example_test_env.reset()
	for i in range(3):
		obs, reward, done = example_test_env.step(np.ones(example_test_env.num_gen))
	assert done

def test_ens(example_test_env):
	example_test_env.reset()
	example_test_env.step(np.zeros(5))
	assert example_test_env.ens

def test_carbon_cost():
	env = make_env(usd_per_kgco2=0.1)
	env.reset()
	obs, reward, done = env.step(np.ones(5))
	assert np.isclose(env.carbon_cost, 14012.776128632971), "carbon cost was: {}".format(env.carbon_cost)
