#!/usr/bin/env python3

import numpy as np

def cap_status(status, env):
    return np.clip(status, -env.t_min_down, env.t_min_up)

def normalise_capped_status(status, env):
    x_min = -env.t_min_down
    x_max = env.t_min_up
    return 2*(status - x_min) / (x_max - x_min) - 1

def cap_and_normalise_status(status, env):
    return normalise_capped_status(cap_status(status, env), env)

def process_observation(obs, env, forecast_errors=False):
    """
    Process an observation, normalising variables where possible.
    """
    status_norm = cap_and_normalise_status(obs['status'], env)
    demand_norm = obs['demand_forecast']/env.max_demand
    wind_norm = obs['wind_forecast']/env.max_demand
    demand_errors_norm = obs['demand_errors']/env.max_demand
    wind_errors_norm = obs['wind_errors']/env.max_demand
    timestep_norm = (obs['timestep'])/env.episode_length

    if forecast_errors:
        processed_obs = np.concatenate((status_norm,
                                       demand_norm,
                                       wind_norm,
                                       demand_errors_norm,
                                       wind_errors_norm))
    else:
        processed_obs = np.concatenate((status_norm,
                                       demand_norm,
                                       wind_norm))

    return processed_obs
