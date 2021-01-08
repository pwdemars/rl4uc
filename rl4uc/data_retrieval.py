#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import requests
import io
import dotenv

SEED = 2

def get_wind_data(start_date, end_date, unit_id):

	COLUMNS = ['Settlement Date', 'SP', 'Quantity (MW)'] # specify only these required columns

	date_range = pd.date_range(start_date, end_date).tolist()
	date_range = [f.strftime('%Y-%m-%d') for f in date_range]
	all_df = pd.DataFrame(columns=COLUMNS)

	for date in date_range: 
	    url = 'https://api.bmreports.com/BMRS/B1610/v2?APIKey={}&SettlementDate={}&Period=*&ServiceType=csv&NGCBMUnitID={}'.format(API_KEY, date, unit_id)
	    response = requests.get(url, allow_redirects=True)

	    # painful way of checking for an empty response...
	    if 'Content-Disposition' not in response.headers:
	    	print("Could not read {}".format(date))
	    	continue 

	    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), header=1).filter(COLUMNS)
	    all_df = all_df.append(df)

	# Sort and rename cols
	all_df = all_df.sort_values(['Settlement Date', 'SP']).reset_index(drop=True)
	all_df = all_df.rename(columns={'Settlement Date': 'date', 'SP': 'period', 'Quantity (MW)': 'wind'})
	all_df.date = pd.to_datetime(all_df.date)

	return all_df 

def get_demand_data(start_date, end_date):
	date_range = pd.date_range(start_date, end_date).tolist()
	date_range = [f.strftime('%Y-%m-%d') for f in date_range]

	COLUMNS = ['date', 'period', 'demand']
	all_df = pd.DataFrame(columns=COLUMNS)

	for date in date_range:
		url = 'https://api.bmreports.com/BMRS/SYSDEM/v1?APIKey={}&FromDate={}&ToDate={}&ServiceType=csv'.format(API_KEY, date, date)
		response = requests.get(url, allow_redirects=True)
		df = pd.read_csv(io.StringIO(response.content.decode('utf-8'))).reset_index()

		# Rename columns
		df = df.rename(columns={'level_0': 'type', 'level_1': 'date', 'HDR': 'period', 'SYSTEM DEMAND': 'demand'})

		# Get just ITSDO standing for Initial Transmission System Demand Outturn
		df = df[df.type=='ITSDO']

		# Reformat dates and periods
		df.date = pd.to_datetime(df.date, format='%Y%m%d')
		df.period = df.period.astype('int')

		df = df.filter(COLUMNS)

		all_df = all_df.append(df)

	all_df.date = pd.to_datetime(all_df.date)

	return all_df

if __name__=="__main__":

	dotenv.load_dotenv('../.env')
	API_KEY = os.getenv('BMRS_KEY')

	SAVE_DIR = 'data'
	os.makedirs(SAVE_DIR, exist_ok=True)

	WIND_UNIT_ID = 'WHILW-1' # specify the BM Unit ID to retrieve for wind data
	N_TESTS = 50

	# Dates for entire data set
	start_date = '2016-01-01'
	end_date = '2019-12-31'

	# Train/test split dates ([0] is beginning [1] is end)
	# train_dates = ('2016-01-01', '2018-12-31')
	# test_dates = ('2019-01-01', '2019-12-31')

	print("Getting wind data...")
	wind_df = get_wind_data(start_date, end_date, WIND_UNIT_ID)
	print("Getting demand data...")
	demand_df = get_demand_data(start_date, end_date)

	# Merge demand and wind dfs
	all_df = pd.merge(demand_df, wind_df, on=['date', 'period'])
	
	# Ensure that all days are complete and have no nas 
	all_df = all_df.dropna()
	all_df = all_df.groupby('date').filter(lambda x: len(x) == 48)

	# Now we scale both demand and wind to fit the 10 generator problem. 
	gen_info = pd.read_csv('data/kazarlis_units_10.csv')
	max_demand = sum(gen_info.max_output)
	min_demand = sum(gen_info.max_output) * 0.4
	all_df.demand = (all_df.demand - min(all_df.demand)) / (np.ptp(all_df.demand))
	all_df.demand = all_df.demand * (max_demand - min_demand) + min_demand

	# Scale wind 
	max_wind = sum(gen_info.max_output) * 0.4
	min_wind = 0
	all_df.wind = (all_df.wind - min(all_df.wind)) / np.ptp(all_df.wind)
	all_df.wind = all_df.wind * (max_wind - min_wind) + min_wind

	# # Now we need to scale wind such that it meets the desired wind penetration throughout the data set 
	# # Scale linearly by a factor of target total wind / current total wind 
	# current_wind = sum(all_df.wind) 
	# target_wind = sum(all_df.demand) * WIND_PEN
	# all_df.wind = all_df.wind * target_wind / current_wind 

	# Split train and test data
	test_days = np.random.choice(pd.unique(all_df.date), N_TESTS, replace=False)
	train_df = all_df[~all_df.date.isin(test_days)]
	test_df = all_df[all_df.date.isin(test_days)]

	# train_df = all_df[(all_df.date >= train_dates[0]) & (all_df.date <= train_dates[1])]
	# test_df = all_df[(all_df.date >= test_dates[0]) & (all_df.date <= test_dates[1])]
	
	# Check that wind never exceeds demand. 
	assert(all(all_df.demand > all_df.wind)), "demand exceeds wind at some timesteps. turn down wind penetration"

	# Save to .csv
	train_df.to_csv(os.path.join(SAVE_DIR, 'train_data_10gen.csv'), index=False)
	test_df.to_csv(os.path.join(SAVE_DIR, 'test_data_10gen.csv'), index=False)

