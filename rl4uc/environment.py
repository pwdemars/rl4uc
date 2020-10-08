#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import os
from scipy.stats import norm

from .dispatch import lambda_iteration, calculate_costs
from .generate_demand import scale_demand

DEFAULT_VOLL=1000
DEFAULT_EPISODE_LENGTH=336
DEFAULT_DISPATCH_RESOLUTION=0.5
DEFAULT_DISPATCH_FREQ_MINS=30
DEFAULT_UNCERTAINTY_PARAM=0.
DEFAULT_MIN_REWARD_SCALE=-3000
DEFAULT_NUM_GEN=5
DEFAULT_GAMMA=1.0
DEFAULT_DEMAND_UNCERTAINTY = 0.0
DEFAULT_EXCESS_CAPACITY_PENALTY_FACTOR = 2e3

class Env(object):
    """
    Environment class holding information about the grid state, the demand 
    forecast, the generator information. Methods include calculating costs of
    actions; advancing grid in response to actions. 
    
    TODO: dispatchable wind
    """
    def __init__(self, gen_info, forecast, forecast_norm, mode='train', **kwargs):
        
        modes = ['test', 'train']
        if mode not in modes:
            raise ValueError("Invalid mode: must be test or train")
        else:
            self.mode = mode # Test or train. Determines the reward function and is_terminal()
        
        self.gen_info = gen_info
        self.all_forecast = forecast
        self.all_forecast_norm = forecast_norm
        self.voll = kwargs.get('voll', DEFAULT_VOLL)
        self.scale = kwargs.get('uncertainty_param', DEFAULT_UNCERTAINTY_PARAM)
        self.dispatch_freq_mins = kwargs.get('env_dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS) # Dispatch frequency in minutes 
        self.dispatch_resolution = self.dispatch_freq_mins/60.
        self.num_gen = self.gen_info.shape[0]
        if self.mode == 'test':
            # 1 less than length of demand since at hour 0 demand is 
            self.episode_length = len(forecast)
        else:
            self.episode_length = kwargs.get('episode_length', DEFAULT_EPISODE_LENGTH)
            
        # Min reward is a function of number of generators and episode length
        self.min_reward = (kwargs.get('min_reward_scale', DEFAULT_MIN_REWARD_SCALE) *
                           self.num_gen *
                           self.dispatch_resolution) 
        self.gamma = kwargs.get('gamma', DEFAULT_GAMMA)
        self.demand_uncertainty = kwargs.get('demand_uncertainty', DEFAULT_DEMAND_UNCERTAINTY)
        
        # Parameters for ARMA
        # self.arma_sigma = 1*np.mean(forecast)/100
        self.last_error = 0.
        self.arma_z = 0
        self.arma_alpha = 0.99
        self.arma_beta = 0.1
        self.arma_sigma = self.calculate_arma_sigma(self.arma_alpha, self.arma_beta, 
                                                    sum(self.gen_info.max_output)/10, 
                                                    1-0.001)
        
        self.excess_capacity_penalty_factor = (self.num_gen * 
                                               kwargs.get('excess_capacity_penalty_factor', 
                                                               DEFAULT_EXCESS_CAPACITY_PENALTY_FACTOR) *
                                               self.dispatch_resolution)
        
        # Generator info 
        self.max_output = self.gen_info['max_output'].to_numpy()
        self.min_output = self.gen_info['min_output'].to_numpy()
        self.status = self.gen_info['status'].to_numpy()
        self.a = self.gen_info['a'].to_numpy()
        self.b = self.gen_info['b'].to_numpy()
        self.c = self.gen_info['c'].to_numpy()
        self.t_min_down = self.gen_info['t_min_down'].to_numpy()
        self.t_min_up = self.gen_info['t_min_up'].to_numpy()
        self.t_max_up = self.gen_info['t_max_up'].to_numpy()
        self.hot_cost = self.gen_info['hot_cost'].to_numpy()
        self.cold_cost = self.gen_info['cold_cost'].to_numpy()
        self.cold_hrs = self.gen_info['cold_hrs'].to_numpy()
        
        # Min and max demand for clipping demand profiles
        self.min_demand = np.min(self.min_output)
        self.max_demand = np.sum(self.max_output)
        
        self.forecast_length = kwargs.get('forecast_length', max(self.t_min_down))
        
        self.dispatch_tolerance = 1 # epsilon for lambda iteration.
        
        # Calculate heat rates (the cost per MWh at max output for each generator)
        self.heat_rates = (self.a*(self.max_output**2) + self.b*self.max_output + self.c)/self.max_output
        self.gen_info['heat_rates'] = self.heat_rates
        
        self.state = None
        self.forecast = None
        self.forecast_norm = None
        self.start_cost = 0
        self.infeasible=False
        
    def determine_constraints(self):
        """
        Determine which generators must be kept on or off for the next time period.
        """
        self.must_on = np.array([True if 0 < self.status[i] < self.t_min_up[i] else False for i in range(self.num_gen)])
        self.must_off = np.array([True if -self.t_min_down[i] < self.status[i] < 0 else False for i in range(self.num_gen)])
        
    def legalise_action(self, action):
        """
        Convert an action to be legal (remaining the same if it is already legal).
        
        Considers constraints set in self.determine_constraints()
        """
        x = np.logical_or(np.array(action), self.must_on)
        x = x * np.logical_not(self.must_off)
        return(np.array(x, dtype=int))
        
    def get_heat_rates(self, gen_info):  
        """
        Calculate the heat rates ($/MWh) at max output for each generator in gen_info. 
        """ 
        num_gen = len(gen_info)
        a = gen_info['a'].to_numpy()
        b = gen_info['b'].to_numpy()
        c = gen_info['c'].to_numpy()
        outputs = np.array(gen_info['max_output'][:num_gen])
        pl = (a*(outputs**2) + b*outputs + c)/outputs
        return pl
    
    def calculate_arma_sigma(self, alpha, beta, x, p):
        """
        Calculate the standard deviation for the white noise used in the ARMA process.
    
        Parameters
        ----------
        alpha : float
            Coefficient for AR term of ARMA process
        beta : float
            Coefficient for MA term of ARMA process
        x : float
            Value for quantile at p (e.g. reserve constraint)
        p : float (0<p<1)
            Quantile for value x (e.g. probability of error exceeding x)
    
        Returns
        -------
        float
            Standard deviation (sigma) of ARMA process with specified parameters.
    
        """
        num = np.square(x/norm.ppf(p))
        denom = 1 + np.square(alpha + beta)/(1-np.square(alpha))
        return np.sqrt(num/denom)
    
    def sample_demand(self):
        """ 
        Sample a realisation of demand. 
        
        Change the demand distribution by editing this function. 
        """
        return np.clip(self.forecast * np.random.normal(1, self.demand_uncertainty),
                       self.min_demand,
                       self.max_demand)
    
    def sample_error(self, x_last, z_last):
        """
        Sample a realisation of demand forecast error using first order 
        auto-regressive moving average. 
       
        Args:
          - x (float): previous forecast error
          - z (float): previous random noise
        """
        z = np.random.normal(0, self.arma_sigma)
        error = self.arma_alpha*x_last + z + self.arma_beta*z_last
        
        return error, z
    
    def step(self, action):
        """
        Transition a timestep forward following an action.
        
        The ordering of this function is important. The costs need to be calculated
        AFTER the demand has rolled forward, but BEFORE the grid_state is updated
        and update_gen_status is called (because it uses the grid status of the
        period before the dispatch in order to calculate start costs.
        """
        # Fix constrained generators (if necessary)
        action = self.legalise_action(action)
        
        # Advance demand 
        self.episode_timestep += 1
        self.hour += 1
        self.forecast = self.all_forecast[self.hour]
        self.forecast_norm = self.all_forecast_norm[self.hour]
        
        # Sample demand realisation
        error, z = self.sample_error(self.last_error, self.arma_z)
        self.last_error = error
        self.arma_z = z 
        self.demand_real = self.forecast + error
        self.demand_real = np.clip(self.demand_real, self.min_demand, self.max_demand)
        
        # Calculate start costs 
        self.start_cost = self.calculate_start_costs(action)
        
        # Update generator status
        self.grid_state = action
        self.update_gen_status(action)
        
        # Get available actions
        self.determine_constraints()
        
        # Cap and normalise status
        self.cap_and_normalise_status()

        # Assign state
        self.demand_forecast, self.demand_forecast_norm = self.get_demand_forecast()     
        self.state = {'status': self.status,
                      'status_capped': self.status_capped,
                      'status_norm': self.status_norm,
                      'demand_forecast': self.demand_forecast,
                      'demand_forecast_norm': self.demand_forecast_norm,
                      'forecast_error': self.last_error/self.max_demand}
        
        # Calculate fuel cost and dispatch for the demand realisation 
        self.fuel_cost, self.disp = self.calculate_fuel_cost_and_dispatch(self.demand_real)
        
        # Calculating lost load costs and marking ENS
        diff = abs(self.demand_real - np.sum(self.disp))
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        self.ens_cost = ens_amount*self.voll*self.dispatch_resolution
        self.ens = True if ens_amount > 0 else False
        
        reward = self.get_reward(self.demand_real)
        self.reward = reward
        
        done = self.is_terminal()
        
        return self.state, reward, done

    def step_deterministic(self, action):
        """
        Same as step(), except it follows the forecast exactly.
        
        The ordering of this function is important. The costs need to be calculated
        AFTER the demand has rolled forward, but BEFORE the grid_state is updated
        and update_gen_status is called (because it uses the grid status of the
        period before the dispatch in order to calculate start costs.
        """
        
        # Fix constrained generators (if necessary)
        action = self.legalise_action(action)
        
        # Advance demand 
        self.episode_timestep += 1
        self.hour += 1
        self.forecast = self.all_forecast[self.hour]
        self.forecast_norm = self.all_forecast_norm[self.hour]
        
        # Sample demand realisation
        self.demand_real = self.forecast
        
        # Calculate start costs 
        self.start_cost = self.calculate_start_costs(action)
        
        # Update generator status
        self.grid_state = action
        self.update_gen_status(action)
        
        # Get available actions
        self.determine_constraints()
        
        # Cap and normalise status
        self.cap_and_normalise_status()

        # Assign state
        self.demand_forecast, self.demand_forecast_norm = self.get_demand_forecast()     
        self.state = {'status': self.status,
                      'status_capped': self.status_capped,
                      'status_norm': self.status_norm,
                      'demand_forecast': self.demand_forecast,
                      'demand_forecast_norm': self.demand_forecast_norm,
                      'forecast_error': self.last_error/self.max_demand}
        
        # Calculate fuel cost and dispatch for the demand realisation 
        self.fuel_cost, self.disp = self.calculate_fuel_cost_and_dispatch(self.demand_real)
        
        # Calculating lost load costs and marking ENS
        diff = abs(self.demand_real - np.sum(self.disp))
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        self.ens_cost = ens_amount*self.voll*self.dispatch_resolution
        self.ens = True if ens_amount > 0 else False
        
        reward = self.get_reward(self.demand_real)
        self.reward = reward
        
        done = self.is_terminal()
        
        return self.state, reward, done
    
    def get_reward(self, demand_real):
        """
        Calculate the reward.
        
        If training, the reward is a linear function of min_reward in the case of ENS
        (increasing over the episode length) otherwise it is the negative expected cost. 
        
        If testing, it is always the negative expected cost (lost load is penalised)
        at VOLL.
        """
        # Operating cost is the sum of fuel cost, ENS cost and start cost.
        # Start cost is not variable: it doesn't depend on demand realisation.
        operating_cost = self.fuel_cost + self.ens_cost + self.start_cost
        
        if self.mode == 'train':
            # Spare capacity penalty:
            reserve_margin = np.dot(self.grid_state, self.max_output)/self.forecast - 1
            excess_capacity_penalty = self.excess_capacity_penalty_factor * np.square(max(0,reserve_margin))

            reward = self.min_reward if self.ens else -operating_cost - excess_capacity_penalty
        else: 
            reward = -operating_cost

        return reward
    

    def sample_reward_new(self, x, z):
        """
        Sample a new realisation of demand, with forecast errors following an 
        ARMA model. 

        Args:
          - x (float): previous forecast error
          - z (float): previous white noise
        """
        if self.mode=="train":
            raise ValueError("sample reward is not yet available in training mode")
        
        error, z = self.sample_error(x, z)
        demand_real = self.forecast + error
        demand_real = np.clip(demand_real, self.min_demand, self.max_demand)     
        
        fuel_cost, disp = self.calculate_fuel_cost_and_dispatch(demand_real)

        diff = abs(demand_real - np.sum(disp))
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        ens_cost = ens_amount*self.voll*self.dispatch_resolution
        is_ens = ens_amount > 0
        
        operating_cost = fuel_cost + ens_cost + self.start_cost # Note start cost is invariant with demand

        return -operating_cost, is_ens, error, z
    
    def sample_reward(self):
        """
        Generate a new realisation of demand and calculate reward. 
        
        This effectively calculates the reward for a state s' that is indentical to self
        in all ways except demand realisation. It can therefore be used to estimate 
        expected reward for a (state, action) pair.
        """
        if self.mode == "train":
            raise ValueError("Sample reward is not yet available in training mode")
        
        demand_real = self.sample_demand()
        
        fuel_cost, disp = self.calculate_fuel_cost_and_dispatch(demand_real)
        
        diff = abs(demand_real - np.sum(disp))
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        ens_cost = ens_amount*self.voll*self.dispatch_resolution
        is_ens = ens_amount > 0
        
        operating_cost = fuel_cost + ens_cost + self.start_cost # Note start cost is invariant with demand
        
        return -operating_cost, is_ens
        
    def get_demand_forecast(self):
        """
        Return the absolute and normalised demand forecasts for the next 
        forecast_length periods. Used to update the state attribute. 
        """
        a = self.hour+1
        z = self.end_hour+1
        demand_forecast = self.all_forecast[a:z]
        demand_forecast_norm = self.all_forecast_norm[a:z]
        return demand_forecast, demand_forecast_norm
        
    def update_gen_status(self, action):
        """Subroutine updating generator statuses.""" 
        # TODO think about if this can be vectorised.
        
        def single_update(status, action):
            if status > 0:
                if action == 1:
                    return (status + 1)
                else:
                    return -1
            else:
                if action == 1:
                    return 1
                else:
                    return (status - 1)
        self.status = np.array([single_update(self.status[i], action[i]) for i in range(len(self.status))])
        
    def economic_dispatch(self, action, demand, lambda_lo, lambda_hi):
        """Calcuate economic dispatch using lambda iteration.
        
        Args:
            action (numpy array): on/off commands
            lambda_lo (float): initial lower lambda value
            lambda_hi (float): initial upper lambda value
            
        """
        idx = np.where(np.array(action) == 1)[0]
        on_a = self.a[idx]
        on_b = self.b[idx]
        on_min = self.min_output[idx]
        on_max = self.max_output[idx]
        disp = np.zeros(self.num_gen)
        if np.sum(on_max) < demand:
            econ = on_max
        elif np.sum(on_min) > demand:
            econ = on_min
        else:
            econ = lambda_iteration(demand, lambda_lo,
                                             lambda_hi, on_a, on_b,
                                             on_min, on_max, self.dispatch_tolerance)
        for (i, e) in zip(idx, econ):
            disp[i] = e        
            
        return disp
        
    def calculate_start_costs(self, action):
        """
        Calculate start costs incurred when stepping self with action.
        """
        # Start costs
        idx = [list(map(list, zip(action, self.grid_state)))[i] == [1,0] for i in range(self.num_gen)]
        idx = np.where(idx)[0]
        start_cost = 0
        for i in idx:
            if abs(self.grid_state[i]) <= self.cold_hrs[i]: #####
                start_cost += self.hot_cost[i]
            else:
                start_cost += self.cold_cost[i]
        
        return start_cost

    def calculate_expected_costs(self, action):
        """
        If self.scale == 0 (meaning there is no error in demand forecast) then
        calculate costs for the nominal demand.
        
        Otherwise calculate expected costs with 5 samples: at point forecast,
        and ±1,2 sigma from the forecast.
        
        Costs are calculated using a weighted average of these 5 values: 68.2% 
        for the point forecast, 13.6% for ±sigma, 2.3% for ±2sigma. 
        """
        
        if self.scale == 0:
            return self.calculate_nominal_costs(action)
        
        else: 
            # Weights for sum (based on normal distribution CDF)
            sigmas = [0.023, 0.136, 0.682, 0.136, 0.023]
            
            a = np.tile(self.forecast, 5)
            b = np.array([self.forecast*self.scale*i for i in range(-2,3)])
            demands = a+b
            
        average_cost = 0
        average_ens_cost = 0
        
        for i, d in enumerate(demands):
            fuel_cost, disp = self.calculate_fuel_cost_and_dispatch(action)
            
            # Energy-not-served costs
            diff = abs(d - sum(disp))
            ens = diff if diff > self.dispatch_tolerance else 0
            ens_cost = ens*self.voll*self.dispatch_resolution

            average_cost += sigmas[i]*(fuel_cost + ens_cost)
            average_ens_cost += sigmas[i]*ens_cost
        
        self.expected_cost = average_cost + self.start_cost
        self.ens = True if average_ens_cost > 0 else False
        
    def calculate_fuel_cost_and_dispatch(self, demand):
        """
        Calculate the economic dispatch to meet demand.
        
        Returns:
            - fuel_cost (float)
            - dispatch (array): power output for each generator
        """
        # Get economic dispatcb
        disp = self.economic_dispatch(self.grid_state, demand, 0, 100)
        
        # Calculate fuel costs costs
        fuel_cost = sum(calculate_costs(disp, self.a, self.b, self.c, self.dispatch_resolution))
        
        return fuel_cost, disp
        
    def cap_and_normalise_status(self):
        """
        Transform the environment status by capping at min up/down times, then
        normalise to between -1 (off and available to turn on) and 1 (on and 
        available to turn off). 
        """
        
        # Cap state
        self.status_capped = np.clip(self.status, -self.t_min_down, self.t_min_up)
        
        # Normalise
        x_min = -self.t_min_down
        x_max = self.t_min_up
        self.status_norm = 2*(self.status_capped - x_min) / (x_max - x_min) - 1
        
    def is_feasible(self): 
        """
        Determine whether there is enough capacity to meet nominal demand in 
        current and future periods considering minimum down time constraints.    
        
        This does not consider case where there is not enough footroom to meet 
        demand (min_output constraints are violated): this is typically 
        less common.
        """
        # Infeasible if demand can't be met in current period (except in initial period)
        if self.episode_timestep > 0:
            if np.dot(self.grid_state, self.max_output) < self.forecast:
                return False
        
        # If all generators are on, demand can definitely be met (upwards)
        if np.all(self.grid_state):
            return True
        
        # Determine how many timesteps ahead we need to consider
        horizon = max(0, np.max((self.t_min_down + self.status)[np.where(self.grid_state == 0)])) # Get the max number of time steps required to determine feasibility
        horizon = min(horizon, self.episode_length-self.episode_timestep) # Horizon only goes to end of day
        
        for t in range(horizon):
            demand = self.all_forecast[self.hour+t+1] # Nominal demand for t+1th period ahead
            future_status = self.status + (t+1)*np.where(self.status >0, 1, -1) # Assume all generators are kept on where possible
            
            available_generators = (-future_status >= self.t_min_down) | self.grid_state # Determines the availability of generators as binary array
            available_cap = np.dot(available_generators, self.max_output)
            
            if available_cap < demand:
                return False
        
        # If all of the above is satisfied, return True 
        return True

    def is_terminal(self):
        """
        Determine whether the environment is in a terminal state. 
        
        When training, the state is terminal if there is energy not served or 
        if at the final timestep of the episode. 
        
        When testing, terminal states only occur at the end of the episode. 
        """
        if self.mode == "train":
            return (self.episode_timestep == self.episode_length) or self.ens
        else: 
            return self.episode_timestep == self.episode_length
    
    def reset(self):
        """
        Returns an initial observation. 
        
        - Set episode timestep to 0. 
        - Choose a random hour to start the episode
        - Reset generator statuses
        - Determine constraints
        """
        
        # Initialise timestep and choose random hour to begin episode 
        if self.mode == 'train':
            self.start_hour = self.hour = np.random.choice(len(self.all_forecast) - 2*self.episode_length) # leave some buffer
        else:
            self.start_hour = self.hour = -1 # Set to 1 period before begin of demand profile.
        
        # Set end peiod
        self.end_hour = self.start_hour + self.episode_length
            
        self.episode_timestep = 0
        self.forecast = None
        self.last_error = 0
        self.arma_z = 0
        self.demand_real = self.all_forecast[self.hour]
        
        # Initalise grid status and constraints
        self.status = self.gen_info['status'].to_numpy()
        self.grid_state = np.where(self.status > 0, 1, 0)
        self.determine_constraints()
        
        # Cap and normalise
        self.cap_and_normalise_status()
        
        # Initialise cost and ENS
        self.expected_cost = 0
        self.ens = False

        # Assign state
        self.demand_forecast, self.demand_forecast_norm = self.get_demand_forecast() 
        self.state = {'status': self.status,
                      'status_capped': self.status_capped,
                      'status_norm': self.status_norm,
                      'demand_forecast': self.demand_forecast,
                      'demand_forecast_norm': self.demand_forecast_norm,
                      'forecast_error': 0.}
        
        return self.state

def make_env(mode="train", forecast=None, reference_forecast=None, **params):
    """
    Create an environment. 
    
    The params file must include the number of generators 
    """
    
    valid_gens = [5, 10, 20]
    valid_dispatch_freq_mins = [5, 15, 30]
    
    if params.get('num_gen', DEFAULT_NUM_GEN) not in valid_gens:
        raise ValueError("Invalid number of generators: must be one of: {}".format(valid_gens))
        
    if params.get('env_dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS) not in valid_dispatch_freq_mins:
        raise ValueError("Invalid dispatch frequency: must be one of: {} minutes".format(valid_dispatch_freq_mins))
        
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Get original kazarlis unit data (at 1 hour resolution)
    gen_info_fn = 'data/kazarlis_units_' + str(params.get('num_gen', DEFAULT_NUM_GEN)) + '.csv'
    gen_info = pd.read_csv(os.path.join(script_dir, gen_info_fn))
    
    # Scale constraint times and initial status (in periods) with dispatch frequency
    # E.g. if dispatch frequency is 30 mins, multiply by 2. 
    gen_info.t_min_up = gen_info.t_min_up * (60/params.get('env_dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
    gen_info.t_min_down = gen_info.t_min_down * (60/params.get('env_dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
    gen_info.status = gen_info.status * (60/params.get('env_dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
    gen_info = gen_info.astype({'t_min_down': 'int64',
                                't_min_up': 'int64',
                                'status': 'int64'})
    
    if forecast is None:
        # Default forecast is National Grid 5 years, at 30 mins resolution
        forecast = np.loadtxt(os.path.join(script_dir, 'data/NG_data_5_years.txt'))
            
        # Interpolate forecast
        upsample_factor = int(30/params.get('env_dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
        xp = np.arange(0, forecast.size)*upsample_factor
        x = np.arange(xp[-1])
        forecast = np.interp(x, xp, forecast)
        
        forecast_scaled = scale_demand(forecast, forecast, gen_info)
        forecast_norm = (forecast_scaled - min(forecast_scaled))/np.ptp(forecast_scaled)
    elif (forecast is not None) and (reference_forecast is not None):
        # If both forecast and reference_forecast are given, then first scale reference forecast,
        # then use scaled reference forecast to scale forecast...
        reference_forecast_scaled = scale_demand(reference_forecast, reference_forecast, gen_info)
        forecast_scaled = scale_demand(reference_forecast, forecast, gen_info)
        forecast_norm = (forecast_scaled - min(reference_forecast_scaled))/np.ptp(reference_forecast_scaled)
    elif (forecast is not None) and (reference_forecast is None):
        # This can be used when the normalised forecast is not needed: for instance,
        # when just testing a schedule through the environment (without any policy).
        # In this case the forecast should already be scaled
        forecast_scaled = forecast
        forecast_norm = (forecast_scaled - np.min(forecast_scaled)) / np.ptp(forecast_scaled)
    else:
        raise ValueError("Can't pass `reference_forecast` on its own. Must provide `forecast`")

    env = Env(gen_info, forecast_scaled, forecast_norm, mode, **params)
    env.reset()
            
    return env
