#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import os
import json
from scipy.stats import weibull_min

from .dispatch import lambda_iteration


DEFAULT_PROFILES_FN='data/train_data_10gen.csv'

DEFAULT_VOLL=10000
DEFAULT_EPISODE_LENGTH_HRS=24
DEFAULT_DISPATCH_RESOLUTION=0.5
DEFAULT_DISPATCH_FREQ_MINS=30
DEFAULT_MIN_REWARD_SCALE=-5000
DEFAULT_NUM_GEN=5
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

def update_cost_coefs(gen_info, usd_per_kgco2):
    factor = (gen_info.kgco2_per_mmbtu / gen_info.usd_per_mmbtu) * usd_per_kgco2
    gen_info.a *= (1 + factor)
    gen_info.b *= (1 + factor)
    gen_info.c *= (1 + factor)

class Env(object):
    """ 
    Simulation environment for the UC problem.

    Methods include calculating costs of actions; advancing grid in response to actions. 
    """
    def __init__(self, gen_info, profiles_df,
                 mode='train', **kwargs):

        self.mode = mode # Test or train. Determines the reward function and is_terminal()
        self.gen_info = gen_info
        self.profiles_df = profiles_df
        
        self.voll = kwargs.get('voll', DEFAULT_VOLL) # value of lost load
        self.dispatch_freq_mins = kwargs.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS) # Dispatch frequency in minutes 
        self.dispatch_resolution = self.dispatch_freq_mins/60.
        self.num_gen = self.gen_info.shape[0]
        if self.mode == 'test':
            self.episode_length = len(self.profiles_df)
        else:
            self.episode_length = kwargs.get('episode_length_hrs', DEFAULT_EPISODE_LENGTH_HRS)
            self.episode_length = int(self.episode_length * (60 / self.dispatch_freq_mins))

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

        # Update the quadratic cost curves to account for carbon price 
        update_cost_coefs(self.gen_info, float(kwargs.get('usd_per_kgco2', 0.)))

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
        self.mmbtu_per_mwh = 3.412
        
        # Min and max demand for clipping demand profiles
        self.min_demand = np.max(self.min_output)
        self.max_demand = np.sum(self.max_output) 
        
        # Tolerance parameter for lambda-iteration 
        self.dispatch_tolerance = 1 # epsilon for lambda iteration.

        # Max cost per mwh
        self.gen_info['max_cost_per_mwh'] = (self.a*(self.min_output**2) + self.b*self.min_output + self.c)/self.min_output

        # Carbon emissions data 
        self.usd_per_kgco2 = float(kwargs.get('usd_per_kgco2', 0.)) # $20 per tonne 
        
        self.forecast = None
        self.start_cost = 0
        self.infeasible=False
        self.day_cost = 0 # cost for the entire day 

        # Min reward is a function of number of generators and episode length
        self.min_reward = (kwargs.get('min_reward_scale', DEFAULT_MIN_REWARD_SCALE) *
                           self.num_gen *
                           self.dispatch_resolution * 
                           (1 + self.usd_per_kgco2 * 20)) 

        # Calculate minimum fuel cost per MWh
        self.min_fuel_cost = (self.a*(self.max_output**2) + self.b*self.max_output + self.c)/self.max_output
        self.gen_info['min_fuel_cost'] = self.min_fuel_cost

        # Set up for outages
        self.outages = kwargs.get('outages', False)
        if kwargs.get('outages', False):
            self.outage_rate = self.gen_info['outage_rate'].to_numpy()
            self.max_outages = int(self.num_gen)/10
        else:
            self.outage_rate = np.zeros(self.num_gen)
        self._reset_availability()
        self.weibull = False
        self.weibull_loc = 0
        self.weibull_scale = 100

        # Set up for curtailment
        self.curtailment = kwargs.get('curtailment', False)
        self.curtail_size_mw = kwargs.get('curtail_size_mw', 100000)
        self.curtailed_mwh = 0
        self.curtailment_factor = kwargs.get('curtailment_factor', 0.)

        self.action_size = self.num_gen + int(self.curtailment)


    def _reset_availability(self):
        self.availability = np.ones(self.num_gen)

    def _update_availability(self, outage):
        self.availability -= outage
        self.availability = np.clip(self.availability, 0, 1)

    def _sample_outage(self, availability, commitment, status):

        if self.weibull: 
            # note: when status < 0, probability = 0
            probs = weibull_min.pdf(status, self.gen_info.outage_weibull_shape, self.weibull_loc, self.weibull_scale)
        else:
            probs = self.outage_rate

        outage = np.random.binomial(1, probs)
        # Generator experience an outage if: 
        #   1. going from on --> off (status > 0, commitment == 0)
        #   2. going from off --> on (status < 0, commitment == 1), since probs(x<1)=0 (see note above)
        outage = outage * commitment # outages are only possible when generator is already on
        if outage.sum() > 1: # only one outage at a time 
            outage = self._sample_outage(availability, commitment, status)
        return outage
        
    def _determine_constraints(self):
        """
        Determine which generators must be kept on or off for the next time period.
        """
        self.must_on = np.array([True if 0 < self.status[i] < self.t_min_up[i] else False for i in range(self.num_gen)])
        self.must_off = np.array([True if -self.t_min_down[i] < self.status[i] < 0 else False for i in range(self.num_gen)])
        
    def _legalise_action(self, action):
        """
        Convert an action to be legal (remaining the same if it is already legal).
        
        Considers constraints set in self.determine_constraints()
        """
        x = np.logical_or(np.array(action), self.must_on)
        x = x * np.logical_not(self.must_off)
        return(np.array(x, dtype=int))
        
    def _is_legal(self, action):
        """
        Check if an action satisfies minimum up/down time constraints
        """
        action = np.array(action)
        illegal_on = np.any(action[self.must_on] == 0)
        illegal_off = np.any(action[self.must_off] == 1)
        if any([illegal_on, illegal_off]):
            return False
        else:
            return True
    
    def _get_net_demand(self, deterministic, errors, curtail=False):
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

        if self.outages:
            max_demand = np.dot(self.max_output, self.availability)
        else:
            max_demand = self.max_demand

        if curtail:
            x = wind_real
            wind_real, wind_curtailed_mw = wind_real * self.curtailment_factor, x - (x * self.curtailment_factor)
            net_demand = demand_real - wind_real
            self.curtailed_mwh = wind_curtailed_mw * self.dispatch_resolution
        else: 
            net_demand = demand_real - wind_real
            self.curtailed_mwh = 0
        net_demand = np.clip(net_demand, self.min_demand, max_demand)
        return net_demand

    def roll_forecasts(self):
        """
        Roll forecasts forward by one timestep
        """
        self.episode_timestep += 1 
        self.forecast = self.episode_forecast[self.episode_timestep]
        self.wind_forecast = self.episode_wind_forecast[self.episode_timestep]

    def calculate_lost_load_cost(self, net_demand, disp, availability=None):

        if self.outages and (availability is not None):
            net_demand = np.minimum(np.dot(availability, self.max_output), net_demand)

        diff = abs(net_demand - np.sum(disp))
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        ens_cost = ens_amount*self.voll*self.dispatch_resolution
        return ens_cost

    def _get_state(self):
        """
        Get the state dictionary. 
        """
        state = {'status': self.status,
                 'demand_forecast': self.episode_forecast,
                 'demand_errors': self.arma_demand.xs,
                 'wind_forecast': self.episode_wind_forecast,
                 'wind_errors': self.arma_wind.xs,
                 'timestep':self.episode_timestep}
        self.state = state
        return state

    def _get_reward(self):
        """Calculate the reward (negative operating cost)"""
        operating_cost = self.fuel_cost + self.ens_cost + self.start_cost
        reward = -operating_cost

        self.reward=reward

        return reward

    def _transition(self, action, deterministic, errors):

        # Get curtailment action (if using)
        if self.curtailment:
            curtail = bool(action[-1]) # last action 
            commitment_action = np.copy(action)[:-1]
        else:
            curtail = False
            commitment_action = action

        # Check if action is legal and legalise if necessary
        if self._is_legal(commitment_action) is False:
            commitment_action = self._legalise_action(commitment_action)

        # Advance demand 
        self.roll_forecasts()
        
        # Sample demand realisation
        self.net_demand = self._get_net_demand(deterministic, errors, curtail)

        # Sample outages
        if (self.outages and 
            (not deterministic)):
            outage = self._sample_outage(self.availability, commitment_action, self.status)
            self._update_availability(outage)
            
        # Update generator status
        self.commitment = np.array(commitment_action)
        self.update_gen_status(self.commitment)
        
        # Determine whether gens are constrained to remain on/off
        self._determine_constraints()

        # Calculate operating costs
        self.start_cost = self._calculate_start_costs()
        self.fuel_costs, self.disp = self.calculate_fuel_cost_and_dispatch(self.net_demand, commitment_action, self.availability)
        self.fuel_cost = np.sum(self.fuel_costs)
        self.kgco2 = self._calculate_kgco2(self.fuel_costs, self.disp)
        self.ens_cost = self.calculate_lost_load_cost(self.net_demand, self.disp, self.availability)
        self.ens = True if self.ens_cost > 0 else False # Note that this will not mark ENS if VOLL is 0. 

        # Accumulate the total cost for the day
        self.day_cost += self.start_cost + self.fuel_cost + self.ens_cost

        # Assign state
        state = self._get_state()

        return state

    def step(self, action, deterministic=False, errors=None):
        """
        Advance the environment forward 1 timestep, returning an observation, the reward
        and whether the s_{t+1} is terminal. 
        """
        obs = self._transition(action, deterministic, errors) # Advance the environment, return an observation 
        reward = self._get_reward() # Evaluate the reward function 
        done = self.is_terminal() # Determine if state is terminal

        return obs, reward, done
        
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

    def _generator_fuel_costs(self, output, commitment):
        costs = np.multiply(self.a, np.square(output)) + np.multiply(self.b, output) + self.c
        costs = costs * self.dispatch_resolution # Convert to MWh by multiplying by dispatch resolution in hrs
        costs = costs * commitment
        return costs

    def _calculate_fuel_costs(self, output, commitment):
        """ 
        Calculate total fuel costs for each generator, returning the sum.

        The fuel costs are quadratic: C = ax^2 + bx + c
        """
        costs = self._generator_fuel_costs(output, commitment)
        return costs
        
    def _calculate_start_costs(self):
        """
        Calculate start costs
        """
        # Start costs
        idx = np.where(self.status == 1)[0]
        start_cost = np.sum(self.hot_cost[idx]) # only hot costs
        return start_cost
        
    def calculate_fuel_cost_and_dispatch(self, demand, commitment, availability=None):
        """
        Calculate the economic dispatch to meet demand.
        
        Returns:
            - fuel_costs (array)
            - dispatch (array): power output for each generator
        """
        # If using outages, then demand should be capped at availabile generation
        # and commitment should be limited by availability 
        if self.outages and (availability is not None):
            demand = np.minimum(np.dot(availability, self.max_output), demand)
            commitment = commitment * availability

        # Get availability
        disp = self.economic_dispatch(commitment, demand, 0, 100)
        
        # Calculate fuel costs costs
        fuel_costs = self._calculate_fuel_costs(disp, commitment)
        
        return fuel_costs, disp


    def _calculate_kgco2(self, fuel_costs, disp):
        e_out_mmbtu = disp * self.dispatch_resolution * self.mmbtu_per_mwh
        usd_per_mmbtu_out = np.divide(fuel_costs, e_out_mmbtu, where=fuel_costs!=0)
        efficiency = (usd_per_mmbtu_out / 
                      (self.gen_info.usd_per_mmbtu + self.usd_per_kgco2 * self.gen_info.kgco2_per_mmbtu))
        kgco2 = e_out_mmbtu * efficiency * self.gen_info.kgco2_per_mmbtu
        return np.sum(kgco2)
        
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

    def sample_day(self):
        """Sample a random day from self.profiles_df"""
        day = np.random.choice(self.profiles_df.date, 1)
        day_profile = self.profiles_df[self.profiles_df.date == day.item()]
        return day, day_profile
    
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
            day, day_profile = self.sample_day()
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
        self._determine_constraints()
        
        # Initialise cost and ENS
        self.expected_cost = 0
        self.ens = False

        # Assign state
        state = self._get_state()

        self._reset_availability()
        
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

def scale_and_interpolate_profiles(num_gen, 
                                   profiles_df=None, 
                                   target_dispatch_freq=30, 
                                   original_num_gen=10, 
                                   original_dispatch_freq=30):
    """
    Linearly scale demand and wind profiles in profiles_df and interpolate 
    """
    if profiles_df is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        profiles_df = pd.read_csv(os.path.join(script_dir, DEFAULT_PROFILES_FN))

    # Used for interpolating profiles from 30 min to higher resolutions
    # upsample_factor= int(original_dispatch_freq / target_dispatch_freq)

    # profiles_df.demand = interpolate_profile(profiles_df.demand, upsample_factor)
    profiles_df.demand = profiles_df.demand * num_gen/original_num_gen # Linearly scale to number of generators.

    # profiles_df.wind = interpolate_profile(profiles_df.wind, upsample_factor)
    profiles_df.wind = profiles_df.wind * num_gen/original_num_gen
    
    return profiles_df

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
    
        profiles_df.demand = profiles_df.demand * len(gen_info)/10 # Scale up or down depending on number of generators.    
        profiles_df.wind = profiles_df.wind * len(gen_info)/10
    
    if mode == 'test' and profiles_df is None:
        raise ValueError("Must supply demand and wind profiles for testing")

    # Create environment object
    env = Env(gen_info=gen_info, profiles_df=profiles_df, mode=mode, **params)
    env.reset()
    
    return env

def make_env_from_json(env_name='5gen', mode='train', profiles_df=None):
    """
    Create an environment object.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    env_fn = os.path.join(script_dir, 'data/envs/{}.json'.format(env_name))
    params = json.load(open(env_fn))

    gen_info = create_gen_info(params.get('num_gen', DEFAULT_NUM_GEN),
                               params.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
    if mode == 'train':
        if profiles_df is None:
            profiles_df = pd.read_csv(os.path.join(script_dir, DEFAULT_PROFILES_FN))
    
        profiles_df.demand = profiles_df.demand * len(gen_info)/10 # Scale up or down depending on number of generators.
        profiles_df.wind = profiles_df.wind * len(gen_info)/10
    
    if mode == 'test' and profiles_df is None:
        raise ValueError("Must supply demand and wind profiles for testing")

    # Create environment object
    env = Env(gen_info=gen_info, profiles_df=profiles_df, mode=mode, **params)
    env.reset()
    
    return env
