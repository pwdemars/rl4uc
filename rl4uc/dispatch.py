#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:02:35 2018

@author: patrickdemars
"""

import numpy as np
import time

def lambda_iteration(load, lambda_low, lambda_high, a, b, mins, maxs, epsilon):
    """Calculate economic dispatch using lambda iteration. 
    
    lambda_low, lambda_high: initial lower and upper values for lambda
    a: coefficients for quadratic load curves
    b: constants for quadratic load curves
    epsilon: error as a function 
    
    Returns an array of outputs for the generators.
    """    
    num_gen = len(a)
    lambda_low = np.float(lambda_low)
    lambda_high = np.float(lambda_high)
    lambda_mid = 0
    total_output = np.sum(calculate_loads(lambda_high, a, b, mins, maxs, num_gen))
    i = 0 # Counter ensures that this never terminates at the first iteration.
    while abs(total_output - load) > epsilon or i < 1:
        lambda_mid = (lambda_high + lambda_low)/2
        total_output = np.sum(calculate_loads(lambda_mid, a, b, mins, maxs, num_gen))
        if total_output - load > 0:
            lambda_high = lambda_mid
        else:
            lambda_low = lambda_mid
        i += 1

    return calculate_loads(lambda_mid, a, b, mins, maxs, num_gen)

def calculate_loads(lm, a, b, mins, maxs, num_gen):
    """Calculate loads for all generators as a function of lambda.
    lm: lambda
    a, b: coefficients for quadratic curves of the form cost = a^2p + bp + c
    num_gen: number of generators
    
    Returns an array of individual generator outputs. 
    """
    p = (lm - b)/a
    powers = np.where(p < mins, mins, np.where(p > maxs, maxs, p))
    return powers

def calculate_costs(outputs, a, b, c, dispatch_resolution):
    """Calculate production costs. Quadratic cost curves are of the form 
    cost = (a^2(x) + b(x) + c)*time_in_hours
    
    Args:
      - outputs: array of generating outputs
      - a, b, c: arrays of coefficients for quadratic cost curves
      - n_hours: resolution of settlement periods. 0.5 if half-hourly etc. 
    
    Outputs:
        - cost_list: a list of production costs for each unit. 
    """
    num_gen = len(a)
    cost_list = []
    for i in range(num_gen):
        if outputs[i] == 0:
            cost_list.append(0)
        else:      
            cost_unit = dispatch_resolution*(a[i]*(outputs[i]**2) + b[i]*outputs[i] + c[i])
            cost_list.append(cost_unit)
    return cost_list

