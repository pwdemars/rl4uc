#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import os
from scipy.stats import norm

from .dispatch import lambda_iteration
from .generate_demand import scale_demand

DEFAULT_DEMAND_DATA_FN='data/NG_data_5_years.txt'
DEFAULT_WIND_DATA_FN='data/whitelee_train_pre2019.txt'

DEFAULT_VOLL=1000
DEFAULT_EPISODE_LENGTH_HRS=24
DEFAULT_DISPATCH_RESOLUTION=0.5
DEFAULT_DISPATCH_FREQ_MINS=30
DEFAULT_UNCERTAINTY_PARAM=0.
DEFAULT_MIN_REWARD_SCALE=-5000
DEFAULT_NUM_GEN=5
DEFAULT_GAMMA=1.0
DEFAULT_DEMAND_UNCERTAINTY = 0.0
DEFAULT_EXCESS_CAPACITY_PENALTY_FACTOR = 2e3

class ARMAProcess(object):
    """
    First order ARMA process. 
    """
    def __init__(self, alpha, beta, name, sigma=0):
        self.alpha=alpha
        self.beta=beta
        self.name=name
        self.sigma=sigma
        self.x=0. # Last sampled error
        self.z=0. # Last sampled white noise 
    
    def sample_error(self):
        z = np.random.normal(0, self.sigma)
        x = self.alpha*self.x + self.beta*self.z + z
        self.z = z 
        self.x = x
        return x 
    
    def set_sigma(self, x, p):
        """
        Calculate the standard deviation for the white noise used in the ARMA process
        based on a quantile defined by x and p: P(X<x)=p
        
        For instance, this might be useful for determining relaibility criteria:
        e.g. probability of demand exceeding x should be no greater than p. 
        """
        num = np.square(x/norm.ppf(p))
        denom = 1 + np.square(self.alpha + self.beta)/(1-np.square(self.alpha))
        self.sigma=np.sqrt(num/denom)
        

class Env(object):
    """
    Environment class holding information about the grid state, the demand 
    forecast, the generator information. Methods include calculating costs of
    actions; advancing grid in response to actions. 
    
    TODO: wind
    """
    def __init__(self, gen_info, demand_forecast, wind_forecast,
                 mode='train', **kwargs):

        self.mode = mode # Test or train. Determines the reward function and is_terminal()
        self.gen_info = gen_info
        self.all_forecast = demand_forecast
        self.all_wind = wind_forecast
        
        self.voll = kwargs.get('voll', DEFAULT_VOLL)
        self.scale = kwargs.get('uncertainty_param', DEFAULT_UNCERTAINTY_PARAM)
        self.dispatch_freq_mins = kwargs.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS) # Dispatch frequency in minutes 
        self.dispatch_resolution = self.dispatch_freq_mins/60.
        self.num_gen = self.gen_info.shape[0]
        if self.mode == 'test':
            self.episode_length = len(demand_forecast)
        else:
            self.episode_length = kwargs.get('episode_length_hrs', DEFAULT_EPISODE_LENGTH_HRS)
            self.episode_length = int(self.episode_length * (60 / self.dispatch_freq_mins))
            
        # Min reward is a function of number of generators and episode length
        self.min_reward = (kwargs.get('min_reward_scale', DEFAULT_MIN_REWARD_SCALE) *
                           self.num_gen *
                           self.dispatch_resolution) 
        self.gamma = kwargs.get('gamma', DEFAULT_GAMMA)
        self.demand_uncertainty = kwargs.get('demand_uncertainty', DEFAULT_DEMAND_UNCERTAINTY)
        
        # ARMA processes for demand and wind
        self.arma_demand = ARMAProcess(alpha=0.99, beta=0.1, name='demand')
        self.arma_wind = ARMAProcess(alpha=0.95, beta=0.01, name='wind')
        if self.mode == 'train':
            self.arma_demand.set_sigma(x=sum(self.gen_info.max_output)/10, p=0.999)
            self.arma_wind.set_sigma(x=sum(self.gen_info.max_output)/20, p=0.999)
        else:
            if None in [kwargs.get('demand_sigma'), kwargs.get('wind_sigma')]:
                raise ValueError("Must supply sigmas for demand and wind ARMAs when testing")
            else:
                self.arma_demand.sigma = kwargs.get('demand_sigma')
                self.arma_wind.sigma = kwargs.get('wind_sigma')

        print(self.arma_demand.sigma, self.arma_wind.sigma)
        
        # Penalty factor for committing excess capacity, usedi n training reward function 
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
        
        self.forecast = None
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
    
    def get_net_demand(self, deterministic):
        """
        Sample demand and wind realisations to get net demand forecast. 
        """
        # Determine demand realisation 
        if deterministic is False:
            error = self.arma_demand.sample_error()
        else:
            error = 0 
        demand_real = self.forecast + error
        demand_real = max(0, demand_real)
        
        # Wind realisation
        if deterministic is False:
            error = self.arma_wind.sample_error()
        else:
            error = 0 
        wind_real = self.wind_forecast + error
        wind_real = max(0, wind_real)
        
        # Net demand is demand - wind 
        net_demand = demand_real - wind_real
        net_demand = np.clip(net_demand, self.min_demand, self.max_demand)    
        
        return net_demand

    def step(self, action, deterministic=False):
        """
        Transition a timestep forward following an action.
        
        The ordering of this function is important. The costs need to be calculated
        AFTER the demand has rolled forward, but BEFORE the commitment is updated
        and update_gen_status is called (because it uses the grid status of the
        period before the dispatch in order to calculate start costs.
        """
        # Fix constrained generators (if necessary)
        action = self.legalise_action(action)
        
        # Advance demand 
        self.episode_timestep += 1
        self.forecast = self.episode_forecast[self.episode_timestep]
        self.wind_forecast = self.episode_wind_forecast[self.episode_timestep]
        
        # Sample demand realisation
        self.net_demand = self.get_net_demand(deterministic)
        
        # Calculate start costs 
        self.start_cost = self.calculate_start_costs(action)
        
        # Update generator status
        self.commitment = action
        self.update_gen_status(action)
        
        # Get available actions
        self.determine_constraints()
        
        # Cap and normalise status
        self.cap_and_normalise_status()

        # Assign state
        state = self.get_state()
    
        # Calculate fuel cost and dispatch for the demand realisation 
        self.fuel_cost, self.disp = self.calculate_fuel_cost_and_dispatch(self.net_demand)
        
        # Calculating lost load costs and marking ENS
        diff = abs(self.net_demand - np.sum(self.disp))
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        self.ens_cost = ens_amount*self.voll*self.dispatch_resolution
        self.ens = True if ens_amount > 0 else False
        
        reward = self.get_reward(self.net_demand)
        
        done = self.is_terminal()
        
        return state, reward, done
    
    def get_state(self):
        """
        Get the state dictionary. 
        """
        state = {'status': self.status,
                 'status_capped': self.status_capped,
                 'status_norm': self.status_norm,
                 'demand_forecast': self.episode_forecast[self.episode_timestep+1:],
                 'demand_error': self.arma_demand.x/self.max_demand,
                 'wind_forecast': self.episode_wind_forecast[self.episode_timestep+1:],
                 'wind_error': self.arma_wind.x/self.max_demand}
        self.state = state
        return state

    def get_reward(self, net_demand):
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
            reserve_margin = np.dot(self.commitment, self.max_output)/self.forecast - 1
            excess_capacity_penalty = self.excess_capacity_penalty_factor * np.square(max(0,reserve_margin))

            reward = self.min_reward if self.ens else -operating_cost - excess_capacity_penalty
        else: 
            reward = -operating_cost

        self.reward=reward

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
        disp[idx] = econ
            
        return disp

    def calculate_fuel_costs(self, output):
        """ 
        Calculate total fuel costs for each generator, returning the sum.

        The fuel costs are quadratic: c = ax^2 + bx + c
        """
        costs = np.multiply(output, np.square(self.a)) + np.multiply(output, self.b) + self.c
        costs = costs * self.dispatch_resolution # Convert to MWh by multiplying by dispatch resolution in hrs
        costs = np.sum(costs)
        return costs
        
    def calculate_start_costs(self, action):
        """
        Calculate start costs incurred when stepping self with action.
        """
        # Start costs
        idx = [list(map(list, zip(action, self.commitment)))[i] == [1,0] for i in range(self.num_gen)]
        idx = np.where(idx)[0]
        start_cost = 0
        for i in idx:
            if abs(self.commitment[i]) <= self.cold_hrs[i]: #####
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
        disp = self.economic_dispatch(self.commitment, demand, 0, 100)
        
        # Calculate fuel costs costs
        fuel_cost = self.calculate_fuel_costs(disp)
        
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
        if self.episode_timestep >= 0:
            if np.dot(self.commitment, self.max_output) < self.forecast:
                return False
        
        # If all generators are on, demand can definitely be met (upwards)
        if np.all(self.commitment):
            return True
        
        # Determine how many timesteps ahead we need to consider
        horizon = max(0, np.max((self.t_min_down + self.status)[np.where(self.commitment == 0)])) # Get the max number of time steps required to determine feasibility
        horizon = min(horizon, self.episode_length-self.episode_timestep) # Horizon only goes to end of day
        
        for t in range(horizon):
            demand = self.episode_forecast[self.episode_timestep+t] # Nominal demand for t+1th period ahead
            future_status = self.status + (t+1)*np.where(self.status >0, 1, -1) # Assume all generators are kept on where possible
            
            available_generators = (-future_status >= self.t_min_down) | self.commitment # Determines the availability of generators as binary array
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
            return (self.episode_timestep == (self.episode_length-1)) or self.ens
        else: 
            return self.episode_timestep == (self.episode_length-1)
    
    def reset(self):
        """
        Returns an initial observation. 
        
        - Set episode timestep to 0. 
        - Choose a random hour to start the episode
        - Reset generator statuses
        - Determine constraints
        """
        if self.mode == 'train':
            # Initialise timestep and choose random hour to begin episode 
            x = np.random.choice(self.all_forecast.size - 2*self.episode_length) # leave some buffer
            self.episode_forecast = self.all_forecast[x:x+self.episode_length]
            
            x = np.random.choice(self.all_wind.size-2*self.episode_length)
            self.episode_wind_forecast = self.all_wind[x:x+self.episode_length]
            
        else:
            self.episode_forecast = self.all_forecast
            self.episode_wind_forecast = self.all_wind
        
        # Resetting episode variables
        self.episode_timestep = -1
        self.forecast = None
        self.last_error = 0
        self.last_z = 0
        self.net_demand = None
        
        # Initalise grid status and constraints
        self.status = self.gen_info['status'].to_numpy()
        self.commitment = np.where(self.status > 0, 1, 0)
        self.determine_constraints()
        
        # Cap and normalise
        self.cap_and_normalise_status()
        
        # Initialise cost and ENS
        self.expected_cost = 0
        self.ens = False

        # Assign state
        state = self.get_state()
        
        return state
    
def create_gen_info(num_gen, dispatch_freq_mins):
    """
    Create a gen_info data frame for number of generators and dispatch frequency. 
    Created by copying generators kazarlis 10 gen problem. 
    
    The 10 generator problem is for 30 minute resolution data, so min_down_time
    status, and other time-related vars need to be scaled accordingly.
    
    """
    MIN_GENS = 5
    if num_gen < 5: 
        raise ValueError("num_gen should be at least {}".format(MIN_GENS))
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Repeat generators
    gen10 = pd.read_csv(os.path.join(script_dir,
                                     'data/kazarlis_units_10.csv'))
    upper_limit = int(np.floor(num_gen/10) + 1)
    gen_info = pd.concat([gen10]*upper_limit)[:num_gen]
    gen_info = gen_info.sort_index()
    gen_info.reset_index()
    
    # Scale time-related variables
    gen_info.t_min_up = gen_info.t_min_up * (60/dispatch_freq_mins)
    gen_info.t_min_down = gen_info.t_min_down *  (60/dispatch_freq_mins)
    gen_info.status = gen_info.status * (60/dispatch_freq_mins)
    gen_info = gen_info.astype({'t_min_down': 'int64',
                                't_min_up': 'int64',
                                'status': 'int64'})    
    
    return gen_info

def interpolate_profile(profile, upsample_factor):
    """
    Interpolate a demand/renewables profile, upsampling by a factor of 
    upsample_factor
    """
    xp = np.arange(0, profile.size)*upsample_factor
    x = np.arange(xp[-1])
    interpolated = np.interp(x,xp,profile)
    return interpolated

def scale_profile(profile, scale_range):
    """
    Scale data to the range defined by tuple scale_range
    """
    pmax = max(scale_range)
    pmin = min(scale_range)
    profile_norm = (profile-np.min(profile))/np.ptp(profile)
    profile_scaled = profile_norm * (pmax-pmin) + pmin
    return profile_scaled

def process_profile(profile, upsample_factor, scale_range, gen_info):
    """
    Use this to scale, upsample and normalise a profile (demand or wind). 
    
    Returns:
        - profile
        - profile_norm 
    """
    if upsample_factor > 1:
        profile = interpolate_profile(profile, upsample_factor)
    if scale_range is not None:
        profile = scale_profile(profile, scale_range)
    return profile 
        
def make_env(mode='train', demand=None, wind=None, **params):
    """
    Create an environment object.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    gen_info = create_gen_info(params.get('num_gen', DEFAULT_NUM_GEN),
                               params.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
    if mode == 'train':
        # Used for interpolating profiles from 30 min to higher resolutions
        upsample_factor= int(30/params.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
        ### DEMAND ###
        if demand is None: # use default demand
            demand_forecast = np.loadtxt(os.path.join(script_dir, DEFAULT_DEMAND_DATA_FN))
        DEMAND_LOWER = max(np.max(gen_info.min_output)*10/9, np.sum(gen_info.max_output)*0.1)
        DEMAND_UPPER = np.sum(gen_info.max_output)*10/11
        demand_range = (DEMAND_LOWER, DEMAND_UPPER)
        demand_forecast = process_profile(demand_forecast, upsample_factor, demand_range, gen_info)
    
        ### WIND ###
        if wind is None: # use default wind
            wind_forecast = np.loadtxt(os.path.join(script_dir, DEFAULT_WIND_DATA_FN))
        MAX_WIND = sum(gen_info.max_output)/10 # 10% of max capacity 
        wind_range = (0, MAX_WIND)
        wind_forecast = process_profile(wind_forecast, upsample_factor, wind_range, gen_info)
    
    if mode == 'test':
        if demand is None:
            raise ValueError("Must supply mode for testing")
        if wind is None:
            raise ValueError("Must supply wind for testing")
        demand_forecast = demand
        wind_forecast = wind
    
    # Create environment object
    env = Env(gen_info=gen_info, demand_forecast=demand_forecast, 
              wind_forecast=wind_forecast, mode=mode, **params)
    env.reset()
    
    return env

