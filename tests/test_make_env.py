#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from rl4uc.environment import make_env

def test_make_env():
	for g in [5,10,20,30]:
		make_env(num_gen=g)
		