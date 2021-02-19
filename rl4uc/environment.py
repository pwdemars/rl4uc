#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import os

from .dispatch import lambda_iteration

DEFAULT_PROFILES_FN='data/train_data_10gen.csv'

DEFAULT_VOLL=10000
DEFAULT_EPISODE_LENGTH_HRS=24
DEFAULT_DISPATCH_RESOLUTION=0.5
DEFAULT_DISPATCH_FREQ_MINS=30
DEFAULT_UNCERTAINTY_PARAM=0.
DEFAULT_MIN_REWARD_SCALE=-5000
DEFAULT_NUM_GEN=5
DEFAULT_GAMMA=1.0
DEFAULT_DEMAND_UNCERTAINTY = 0.0
DEFAULT_EXCESS_CAPACITY_PENALTY_FACTOR = 0
DEFAULT_STARTUP_MULTIPLIER=1

DEFAULT_ARMA_PARAMS={"p":5,
                     "q":5,   
                     "alphas_demand":[0.63004456, 0.23178044, 0.08526726, 0.03136807, 0.01153967],
                     "alphas_wind":[0.63004456, 0.23178044, 0.08526726, 0.03136807, 0.01153967],
                     "betas_demand":[0.06364086, 0.02341217, 0.00861285, 0.00316849, 0.00116562],
                     "betas_wind":[0.06364086, 0.02341217, 0.00861285, 0.00316849, 0.00116562],
                     "sigma_demand":10,
                     "sigma_wind":6}

class NStepARMA(object):
    """
    ARMA(N,N) process. May be used for demand or wind. 
    """
    def __init__(self, p, q, alphas, betas, sigma, name):
        self.p=p
        self.q=q
        self.alphas=alphas
        self.betas=betas
        self.name=name
        self.sigma=sigma
        self.xs=np.zeros(p) # last N errors
        self.zs=np.zeros(q) # last N white noise samples

    def sample_error(self):
        zt = np.random.normal(0, self.sigma)
        xt = np.sum(self.alphas * self.xs) + np.sum(self.betas * self.zs) + zt
        return xt, zt

    def step(self, errors=None):
        """
        Step forward the arma process. Can take errors, a (xt, zt) tuple to move this forward deterministically. 
        """
        if errors is not None:
            xt, zt = errors # If seeding
        else:
            xt, zt = self.sample_error()
        self.xs = np.roll(self.xs, 1)
        self.zs = np.roll(self.zs, 1)
        if self.p>0:
            self.xs[0] = xt
        if self.q>0:
            self.zs[0] = zt

        return xt
    
    def reset(self):
        self.xs = np.zeros(self.p)
        self.zs = np.zeros(self.q)


class Env(object):
    """
    Environment class holding information about the grid state, the demand 
    forecast, the generator information. Methods include calculating costs of
    actions; advancing grid in response to actions. 
    
    TODO: wind
    """
    def __init__(self, gen_info, profiles_df,
                 mode='train', **kwargs):

        self.mode = mode # Test or train. Determines the reward function and is_terminal()
        self.gen_info = gen_info
        self.profiles_df = profiles_df
        # self.all_forecast = demand_forecast
        # self.all_wind = wind_forecast
        
        self.voll = kwargs.get('voll', DEFAULT_VOLL)
        self.scale = kwargs.get('uncertainty_param', DEFAULT_UNCERTAINTY_PARAM)
        self.dispatch_freq_mins = kwargs.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS) # Dispatch frequency in minutes 
        self.dispatch_resolution = self.dispatch_freq_mins/60.
        self.num_gen = self.gen_info.shape[0]
        if self.mode == 'test':
            self.episode_length = len(self.profiles_df)
        else:
            self.episode_length = kwargs.get('episode_length_hrs', DEFAULT_EPISODE_LENGTH_HRS)
            self.episode_length = int(self.episode_length * (60 / self.dispatch_freq_mins))
            
        # Min reward is a function of number of generators and episode length
        self.min_reward = (kwargs.get('min_reward_scale', DEFAULT_MIN_REWARD_SCALE) *
                           self.num_gen *
                           self.dispatch_resolution) 
        self.gamma = kwargs.get('gamma', DEFAULT_GAMMA)
        self.demand_uncertainty = kwargs.get('demand_uncertainty', DEFAULT_DEMAND_UNCERTAINTY)

        # Set up the ARMA processes.
        arma_params = kwargs.get('arma_params', DEFAULT_ARMA_PARAMS)
        self.arma_demand = NStepARMA(p=arma_params['p'],
                                     q=arma_params['q'],
                                     alphas=arma_params['alphas_demand'],
                                     betas=arma_params['betas_demand'],
                                     sigma=arma_params['sigma_demand'],
                                     name='demand')
        self.arma_wind = NStepARMA(p=arma_params['p'],
                                   q=arma_params['q'],
                                   alphas=arma_params['alphas_wind'],
                                   betas=arma_params['betas_wind'],
                                   sigma=arma_params['sigma_wind'],
                                   name='wind')
        
        # Penalty factor for committing excess capacity, usedi n training reward function 
        self.excess_capacity_penalty_factor = (self.num_gen * 
                                               kwargs.get('excess_capacity_penalty_factor', 
                                                               DEFAULT_EXCESS_CAPACITY_PENALTY_FACTOR) *
                                               self.dispatch_resolution)

        if mode == 'train':
            # Startup costs are multiplied by this factor in the reward function. 
            self.startup_multiplier = kwargs.get('startup_multiplier', DEFAULT_STARTUP_MULTIPLIER)
        else:
            self.startup_multiplier = 1

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
        self.min_demand = np.max(self.min_output)
        self.max_demand = np.sum(self.max_output) 
        
        self.forecast_length = kwargs.get('forecast_length', max(self.t_min_down))
        
        self.dispatch_tolerance = 1 # epsilon for lambda iteration.
        
        # Calculate heat rates (the cost per MWh at max output for each generator)
        self.heat_rates = (self.a*(self.max_output**2) + self.b*self.max_output + self.c)/self.max_output
        self.gen_info['heat_rates'] = self.heat_rates

        # Max cost per mwh
        self.gen_info['max_cost_per_mwh'] = (self.a*(self.min_output**2) + self.b*self.min_output + self.c)/self.min_output
        
        self.forecast = None
        self.start_cost = 0
        self.infeasible=False
        self.day_cost = 0 # cost for the entire day 
        
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
    
    def is_legal(self, action):
        """
        Check if an action satisfies minimum up/down time constraints
        """
        action = np.array(action)
        illegal_on = np.any(action[self.must_on] == 0)
        illegal_off = np.any(action[self.must_off] == 1)
        if any([illegal_on, illegal_off]):
            print("Illegal action")
            print("Action: {}".format(action))
            print("Must on: {}".format(self.must_on))
            print("Must off: {}".format(self.must_off))
            return False
        else:
            return True

    
    def get_net_demand(self, deterministic, errors):
        """
        Sample demand and wind realisations to get net demand forecast. 
        """
        if errors is not None:
            demand_error = self.arma_demand.step(errors['demand'])
            wind_error = self.arma_wind.step(errors['wind'])

        elif deterministic is True:
            demand_error = wind_error = 0

        else:
            demand_error = self.arma_demand.step()
            wind_error = self.arma_wind.step()

        demand_real = self.forecast + demand_error
        demand_real = max(0, demand_real)
        self.demand_real = demand_real

        wind_real = self.wind_forecast + wind_error
        wind_real = max(0, wind_real)
        self.wind_real = wind_real

        net_demand = demand_real - wind_real
        net_demand = np.clip(net_demand, self.min_demand, self.max_demand)
        return net_demand

    def roll_forecasts(self):
        """
        Roll forecasts forward by one timestep
        """
        self.episode_timestep += 1 
        self.forecast = self.episode_forecast[self.episode_timestep]
        self.wind_forecast = self.episode_wind_forecast[self.episode_timestep]

    def calculate_lost_load_cost(self, net_demand, disp):
        diff = abs(net_demand - np.sum(disp))
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        ens_cost = ens_amount*self.voll*self.dispatch_resolution
        return ens_cost

    def get_state(self):
        """
        Get the state dictionary. 
        """
        state = {'status': self.status,
                 'status_capped': self.status_capped,
                 'status_norm': self.status_norm,
                 'demand_forecast': self.episode_forecast[self.episode_timestep+1:],
                 'demand_errors': self.arma_demand.xs/self.max_demand,
                 'wind_forecast': self.episode_wind_forecast[self.episode_timestep+1:],
                 'wind_errors': self.arma_wind.xs/self.max_demand,
                 'timestep_norm':self.episode_timestep/self.episode_length}
        self.state = state
        return state

    def get_reward(self):
        """
        Calculate the reward.
        
        The reward function may differ between training and test modes. 
        """
        if self.mode == 'train':
            operating_cost = self.fuel_cost + self.ens_cost + self.startup_multiplier*self.start_cost # Apply startup multiplier in training only

            # # Spare capacity penalty:
            # reserve_margin = np.dot(self.commitment, self.max_output)/(self.forecast - self.wind_forecast) - 1
            # excess_capacity_penalty = self.excess_capacity_penalty_factor * np.square(max(0,reserve_margin))

            # reward = self.min_reward if self.ens else -operating_cost - excess_capacity_penalty

            # Reward function that is same as test version: 
            reward = -operating_cost

            # DAILY REWARD FUNCTION
            # if self.is_terminal():
            #     reward = -self.day_cost
            # else:
            #     reward = 0

        else: 
            operating_cost = self.fuel_cost + self.ens_cost + self.start_cost
            reward = -operating_cost

        self.reward=reward

        return reward

    def transition(self, action, deterministic, errors):
        # Check if action is legal
        if self.is_legal(action) is False:
            print("ILLEGAL")

        # Advance demand 
        self.roll_forecasts()
        
        # Sample demand realisation
        self.net_demand = self.get_net_demand(deterministic, errors)
        
        # Update generator status
        self.commitment = np.array(action)
        self.update_gen_status(action)
        
        # Determine whether gens are constrained to remain on/off
        self.determine_constraints()
        
        # Cap and normalise status
        self.cap_and_normalise_status()

        # Calculate operating costs
        self.start_cost = self.calculate_start_costs()
        self.fuel_cost, self.disp = self.calculate_fuel_cost_and_dispatch(self.net_demand)
        self.ens_cost = self.calculate_lost_load_cost(self.net_demand, self.disp)
        self.ens = True if self.ens_cost > 0 else False # Note that this will not mark ENS if VOLL is 0. 

        # Accumulate the total cost for the day
        self.day_cost += self.start_cost + self.fuel_cost + self.ens_cost

        # Assign state
        state = self.get_state()

        return state

    def step(self, action, deterministic=False, errors=None):
        """
        Advance the environment forward 1 timestep, returning an observation, the reward
        and whether the s_{t+1} is terminal. 
        """
        obs = self.transition(action, deterministic, errors) # Advance the environment, return an observation 
        reward = self.get_reward() # Evaluate the reward function 
        done = self.is_terminal() # Determine if state is terminal

        return obs, reward, done
    
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
        
    def calculate_start_costs(self):
        """
        Calculate start costs
        """
        # Start costs
        idx = np.where(self.status == 1)[0]
        start_cost = np.sum(self.hot_cost[idx]) # only hot costs
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
            if np.dot(self.commitment, self.max_output) < self.net_demand:
                return False
        
        # If all generators are on, demand can definitely be met (upwards)
        if np.all(self.commitment):
            return True
        
        # Determine how many timesteps ahead we need to consider
        horizon = max(0, np.max((self.t_min_down + self.status)[np.where(self.commitment == 0)])) # Get the max number of time steps required to determine feasibility
        horizon = min(horizon, self.episode_length-self.episode_timestep) # Horizon only goes to end of day
        
        for t in range(1, horizon):
            net_demand = self.episode_forecast[self.episode_timestep+t] - self.episode_wind_forecast[self.episode_timestep+t] # Nominal demand for t+1th period ahead
            future_status = self.status + (t)*np.where(self.status >0, 1, -1) # Assume all generators are kept on where possible
            
            available_generators = np.logical_or((-future_status >= self.t_min_down), self.commitment) # Determines the availability of generators as binary array
            available_cap = np.dot(available_generators, self.max_output)
            
            if available_cap < net_demand:
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
            # Choose random day
            day = np.random.choice(self.profiles_df.date, 1)
            day_profile = self.profiles_df[self.profiles_df.date == day.item()]
            self.day = day
            self.episode_forecast = day_profile.demand.values
            self.episode_wind_forecast = day_profile.wind.values

        else:
            self.episode_forecast = self.profiles_df.demand.values
            self.episode_wind_forecast = self.profiles_df.wind.values
        
        # Resetting episode variables
        self.episode_timestep = -1
        self.forecast = None
        self.net_demand = None
        self.day_cost = 0
        
        # Reset ARMAs 
        self.arma_demand.reset()
        self.arma_wind.reset()
        
        # Initalise grid status and constraints
        if self.mode == "train": 
            min_max = np.array([self.t_min_down, -self.t_min_up]).transpose()
            self.status = np.array([x[np.random.randint(2)] for x in min_max])
        else:
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
    if num_gen == 5:
        gen_info = gen10[::2] # Special: for 5 gens, take every other generator
    else: 
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
    if upsample_factor == 1:
        return profile
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
        
def make_env(mode='train', profiles_df=None, **params):
    """
    Create an environment object.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    gen_info = create_gen_info(params.get('num_gen', DEFAULT_NUM_GEN),
                               params.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
    if mode == 'train':
        if profiles_df is None:
            profiles_df = pd.read_csv(os.path.join(script_dir, DEFAULT_PROFILES_FN))

        # Used for interpolating profiles from 30 min to higher resolutions
        upsample_factor= int(30/params.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
    
        profiles_df.demand = interpolate_profile(profiles_df.demand, upsample_factor)
        profiles_df.demand = profiles_df.demand * len(gen_info)/10 # Scale up or down depending on number of generators.
    
        profiles_df.wind = interpolate_profile(profiles_df.wind, upsample_factor)
        profiles_df.wind = profiles_df.wind * len(gen_info)/10
    
    if mode == 'test' and profiles_df is None:
        raise ValueError("Must supply demand and wind profiles for testing")

    # Create environment object
    env = Env(gen_info=gen_info, profiles_df=profiles_df, mode=mode, **params)
    env.reset()
    
    return env

