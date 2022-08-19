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

def test_arma_demand_initialisation(example_test_env):
	example_test_env.reset()
	assert np.all(example_test_env.arma_demand.xs == 0)

def test_arma_wind_initialisation(example_test_env):
	example_test_env.reset()
	assert np.all(example_test_env.arma_wind.xs == 0)

def test_arma_sigma_scaling():
	for n in [10, 20, 30]:
		env = make_env(num_gen=n)
		assert env.arma_wind.sigma == 10 * n / 10
		assert env.arma_demand.sigma == 5 * n / 10
