#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np

from rl4uc import helpers
  
class BaseObsProcessor(ABC):
    def __init__(self, env):
        self.env = env
        self.obs_size = len(self.process(env.state))
        super().__init__()
    
    @abstractmethod
    def process(self):
        pass

class DayAheadProcessor(BaseObsProcessor):
    def __init__(self, env, **kwargs):
        self.forecast_errors = kwargs.get('forecast_errors', False)
        BaseObsProcessor.__init__(self, env)        
    
    def process(self, obs, **kwargs):
        """
        Process an observation, normalising variables where possible.
        """
        
        status_norm = helpers.cap_and_normalise_status(obs['status'], self.env)
        
        demand_norm = obs['demand_forecast']/self.env.max_demand
        wind_norm = obs['wind_forecast']/self.env.max_demand
        demand_errors_norm = obs['demand_errors']/self.env.max_demand
        wind_errors_norm = obs['wind_errors']/self.env.max_demand
        timestep_norm = (obs['timestep'])/self.env.episode_length

        if self.forecast_errors:
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

class LimitedHorizonProcessor(BaseObsProcessor):
    def __init__(self, env, **kwargs):
        self.forecast_horizon = kwargs.get('forecast_horizon', 24)
        BaseObsProcessor.__init__(self, env)        
    
    def process(self, obs):
        timestep = obs['timestep']
        status_norm = helpers.cap_and_normalise_status(obs['status'], self.env)
        demand_forecast = obs['demand_forecast'][timestep+1:]
        wind_forecast = obs['wind_forecast'][timestep+1:]

        # Repeat the final value if forecast does not reach the horizon
        if len(demand_forecast) < self.forecast_horizon:
            demand_forecast = np.append(demand_forecast,
                                        np.repeat(demand_forecast[-1],
                                                  self.forecast_horizon-len(demand_forecast)))
        demand_forecast = demand_forecast[:self.forecast_horizon]

        # Repeat the final value if forecast does not reach the horizon
        if len(wind_forecast) < self.forecast_horizon:
            wind_forecast = np.append(wind_forecast,
                                        np.repeat(wind_forecast[-1],
                                                  self.forecast_horizon-len(wind_forecast)))
        wind_forecast = wind_forecast[:self.forecast_horizon]

        # Scale the demand and wind 
        demand_norm = demand_forecast / self.env.max_demand
        wind_norm = wind_forecast / self.env.max_demand

        # Scale the timestep
        timestep_norm = np.array(timestep / self.env.episode_length).reshape(1,)

        processed_obs = np.concatenate((status_norm,
                                        demand_norm,
                                        wind_norm,
                                        timestep_norm))

        return processed_obs