#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:17:35 2019

@author: patrickdemars
"""

import numpy as np

def generate_day(d_0):
    """
    Randomly generates a single day of demand, following a 2 peak cycle.
    """
    
    # Set a random seed
    np.random.seed()
    
    d_5 = np.random.uniform(650, 750)
    d_9 = np.random.uniform(1300, 1500)
    d_15 = np.random.uniform(1000, 1200)
    d_18 = np.random.uniform(1300, 1500)
    d_24 = np.random.uniform(900, 1100)
    
    demand = np.ones(24)
    
    # 00:00 - 05:00
    demand[0] = d_0
    for i in range(1, 5):
        demand[i] = d_0 - i*(d_0 - d_5)/5
    demand[5] = d_5    
    
    # 06:00 - 09:00
    for i in range(6, 9):
        demand[i] = d_5 + (i-5)*(d_9 - d_5)/4
    demand[9] = d_9
    
    # 10:00 - 15:00

    for i in range(10, 15):
        demand[i] = d_9 - (i-8)*(d_9 - d_15)/6
    demand[15] = d_15

    # 16:00 - 18:00
    for i in range(16, 18):
        demand[i] = d_15 + (i-14)*(d_18 - d_15)/3
    demand[18] = d_18
    
    # 19:00 - 23:00
    for i in range(18, 24):
        demand[i] = d_18 - (i-17)*(d_18-d_24)/7
    
    return demand, d_24
    
def generate_demand(num_days, seed=False):
    """
    Generates a continuous demand profile using generate_day() 
    """
    if seed: 
        np.random.seed(1)
    d_0 = np.random.uniform(900, 1100)
    all_demand = np.array([0])
    
    for i in range(num_days):
        day, d_0 = generate_day(d_0)
        all_demand = np.append(all_demand, day)
    
    return all_demand

def scale_demand_old(all_demand, gen_info):
    """Scale a demand profile based on gen_info so that there is 10% footroom
    and 10% headroom and min/max demand.
    
    Based on 2 baseload generators which should never need to turn off.
    """
    
    r_min = all_demand.min() # reference min
    r_max = all_demand.max() # reference max
    
    gen_min = np.array(gen_info.min_output) 
    gen_max = np.array(gen_info.max_output)
    bl_idx = np.argpartition(gen_max, -2)[-2:] # Index of baseload generators
    bl_min = gen_min[bl_idx] # Min outputs of baseload
    min_available = np.sum(bl_min) # min stable generation
    max_available = np.sum(gen_max) # max available generation
    
    t_min = (10/9)*min_available
    t_max = (10/11)*max_available
    
    new_demand = ((all_demand - r_min)/(r_max - r_min))*(t_max - t_min) + t_min
    
    return new_demand

def scale_demand(reference_demand, target_demand, gen_info):
    """
    Scale target_demand such that maximum demand gives 10% headroom, and minimum 
    demand gives 10% footroom.
    
    This ensures that the generators in gen_info can meet deviations in demand
    without switching off baseload (the generator with largest min_output) 
    """
    D_min = np.max(gen_info.min_output) * 10/9 # 10% footroom
    D_max = np.sum(gen_info.max_output) * 10/11 # 10% headroom
    
    new_demand = (target_demand - np.min(reference_demand))/np.ptp(reference_demand)
    new_demand = new_demand * (D_max-D_min) + D_min
    
    return new_demand

    