"""
 ===============================================================================================
 RLab Common Numpy And Pandas Bulk Calculations
 ===============================================================================================
"""

import pandas as pd
import numpy as np
import sys
import time
import math
import traceback
import rlab_common as rlcom

from datetime import timedelta
from matplotlib import pyplot as plt  # import subplots, draw, setp, gca, show
# from matplotlib.figure import Figure
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
from matplotlib import ticker
from sklearn import preprocessing

from fractions import Fraction as F

pd.options.display.max_rows = 10
pd.options.display.max_columns = 15

nan = np.nan
DEBUG = False
VERBOSE = False

TIMEFRAMES = ['DAILY', 'MONTHLY', 'QUARTERLY']


def clean_value_df_col_helper(v):
	""" Cleans fields in a pandas data frame.
	"""
	x = v
	if isinstance(x, str):
		x = x.replace("'..'", "").replace('".."', '')
		x = x.replace("'.'", "").replace('"."', '')
		x = x.replace("'-'", "").replace('"-"', '')
		x = x.replace("'--'", "").replace('"--"', '')
		x = x.replace("--", "").replace('..', '')
		x = x.replace("'.'", "").replace('"."', '')
		x = x.replace('"', '').replace("'", "")
		x = x.replace(' ', '').replace('  ', '').replace(',', '').replace('$', '').replace('%', '')
		if x.__len__() == 1:
			x = x.replace('.', '').replace('-', '').replace(' ', '')
		try:
			if x != '':
				return float(x)
			else:
				return np.nan
		except Exception:
			print(
				'WARNING: Returning NAN | Data frame string column value ' + str(v) +
				', could not be converted to a float via the cleaned value of ' + str(x)
			)
			return np.nan
	elif isinstance(x, complex):
		try:
			x = F().from_float(x.real)
			return x
		except Exception:
			print(
				'WARNING: Returning NAN | Data frame complex column value ' + str(v) +
				', could not be converted to a Fraction via the cleaned value of ' + str(x)
			)
			return np.nan
	else:
		try:
			return float(v)
		except Exception:
			print(
				'WARNING: Data frame col type of ' + str(type(x)) + ', with column value ' + str(v) +
				', could not converted to a float.'
			)
			return np.nan


def clean_value_df_cols(df_in, col):
	""" Cleans fields in a pandas data frame.
	"""
	try:
		df = df_in.copy()
		# Filter
		df = df.ffill()
		df[col] = df[col].apply(lambda x: clean_value_df_col_helper(x))
		df = df.ffill()
		# Convert
		df[col] = df[col].astype(float)
		return df
	except Exception:
		print(
			'ERROR: Could not clean col ' + col + '. Skipping. Field type is ' +
			str(type(df[col].values[0])) +
			' | Dataframe source sample is - \n ' + df_in.tail(5) +
			'\n Value of attempted cleaned column array series as str is - ' +
			str(df[col].values)
		)
		return False


def prefix_df_cols(df_in, prefix):
	""" Prefix data frame column names
	"""
	df = df_in.copy()
	try:
		for col in df.columns:
			skip = ['date']
			is_skip_in_col = False
			for k in skip:
				if k in col:
					is_skip_in_col = True
					break
			if not is_skip_in_col:
				# df = clean_value_df_cols(df, col)
				df = df.rename(columns={col: prefix + '_' + col})
		return df
	except Exception:
		return None


def merge_df(df_a, df_b, na='ffill', **kwargs):
	""" Merge data frames.
	"""
	try:
		# clean first
		dfa = df_a.copy()
		if 'date' not in dfa.columns:
			dfa = dfa.reset_index()
		dfa['date'] = pd.to_datetime(dfa['date'], format='%Y-%m-%d')
		# clean first
		dfb = df_b.copy()
		if 'date' not in dfb.columns:
			dfb = dfb.reset_index()
		dfb['date'] = pd.to_datetime(dfb['date'], format='%Y-%m-%d')
		# merge
		dfa = dfa.copy().set_index('date')
		dfb = dfb.copy().set_index('date')
		df = dfa.merge(dfb, **kwargs)
		# if DEBUG:
		# print('RLCOM: DEBUG: merge_df() | nan df=\n%s' % pd.isna(df.tail(10)))
		df = df.reset_index()
		df = df.sort_values(by='date').copy()
		if na == 'ffill':
			df = df.ffill()
		elif na == 'fillna0':
			df = df.fillna(0)
		elif 'drop' in na:
			df = df.dropna()
		return df.copy()
	except Exception as e:
		if isinstance(df_a, pd.core.frame.DataFrame) and isinstance(df_b, pd.core.frame.DataFrame):
			print(
				'WARNING Could not merge dataframe - \n' + str(df_a.tail(1)) +
				'\nwith dataframe - \n' + str(df_b.tail(1))
			)
		print(
			'ERROR: Exception for merge_df() ERR - \n' + str(traceback.format_exc()) +
			'\n' + str(e)
		)
		return df_a


def get_timeframe_by_dates(date_a, date_b, date_c):
	""" Determines the series period frequency by looking at their dates.
		 We use three dates just incase we start at a weekend.
		 INPUT: date from period 1, date from period 2, date from period 3
		 OUTPUT: monthly, quarterly, daily, weekly, annual
	"""
	a = pd.to_datetime(date_a).strftime('%Y-%m-%d')
	b = pd.to_datetime(date_b).strftime('%Y-%m-%d')
	c = pd.to_datetime(date_c).strftime('%Y-%m-%d')

	a_dt = pd.to_datetime(a)
	b_dt = pd.to_datetime(b)
	c_dt = pd.to_datetime(c)

	diff1 = abs(b_dt - a_dt)
	diff2 = abs(c_dt - b_dt)
	# print('get_timeframe_by_dates() | Difference = ' + str(diff))  # Debug
	two_weeks = timedelta(weeks=2)
	six_weeks = timedelta(weeks=6)
	one_day = timedelta(days=1)
	four_days = timedelta(days=4)
	eleven_days = timedelta(days=11)
	two_months = timedelta(weeks=8)
	four_months = timedelta(weeks=16)
	ten_months = timedelta(weeks=40)
	fourteen_months = timedelta(weeks=60)
	is_a_month = (
		(diff1 > two_weeks and diff1 < six_weeks) or (diff2 > two_weeks and diff2 < six_weeks)
	)
	is_a_quarter = (
		(diff1 > two_months and diff1 < four_months) or (diff2 > two_months and diff2 < four_months)
	)
	is_a_day = (diff1 == one_day) or (diff2 == one_day)
	is_a_week = (
		(diff1 > four_days and diff1 < eleven_days) or (diff2 > four_days and diff2 < eleven_days)
	)
	is_a_year = (
		(diff1 > ten_months and diff1 < fourteen_months) or
		(diff2 > ten_months and diff2 < fourteen_months)
	)
	if is_a_month:
		return 'monthly'
	elif is_a_quarter:
		return 'quarterly'
	elif is_a_day:
		return 'daily'
	elif is_a_week:
		return 'weekly'
	elif is_a_year:
		return 'yearly'
	else:
		return 'ERROR'


def get_timeframe_by_df(df):
	""" Try to auto determine the timeframe of the time series.
	"""
	# Sample the middle data points three times
	df_length = df.count()[0]
	df_middle = int(df_length / 2)
	a = df.tail(df_middle).head(1)['date'].values[0]
	b = df.tail(df_middle + 1).head(1)['date'].values[0]
	c = df.tail(df_middle + 2).head(1)['date'].values[0]
	frame_a = get_timeframe_by_dates(a, b, c)
	a = df.tail(df_middle + 3).head(1)['date'].values[0]
	b = df.tail(df_middle + 4).head(1)['date'].values[0]
	c = df.tail(df_middle + 5).head(1)['date'].values[0]
	frame_b = get_timeframe_by_dates(a, b, c)
	a = df.tail(df_middle + 6).head(1)['date'].values[0]
	b = df.tail(df_middle + 7).head(1)['date'].values[0]
	c = df.tail(df_middle + 8).head(1)['date'].values[0]
	frame_c = get_timeframe_by_dates(a, b, c)
	a_is_b = (frame_a == frame_b)
	b_is_c = (frame_b == frame_c)
	a_is_c = (frame_a == frame_c)
	if a_is_b and b_is_c and a_is_c:
		return frame_a
	if a_is_c:
		return frame_a
	if b_is_c:
		return frame_b
	if a_is_b:
		return frame_a
	if VERBOSE:
		print('WARNING: Time series varies too much. Cannot get its time frame frequency.')
	return False


def get_pd_freq_by_tf(tf):
	""" Convert a time frame into a pandas timeframe value.
	"""
	tf = tf.upper()
	if tf == 'DAILY':
		return 'D'
	elif tf == 'WEEKLY':
		return 'W'
	elif tf == 'MONTHLY':
		return 'M'
	elif tf == 'QUARTERLY':
		return 'Q'
	elif tf == 'YEARLY':
		return 'Y'
	else:
		print('ERROR: get_pd_by_freq() - Invalid timeframe string of %s' % tf)
		sys.exit(1)


def normalise_timeseries_df(df):
	"""Forward fills time series data if there are any skipped data points.
		INPUT: time series df
		OUTPUT: larger time series with filled in data
	"""
	sw = rlcom.StopWatch(title='Normalise|TimeSeries|', debug=DEBUG)
	sw.start()
	try:
		tmpdf = df.sort_values(by='date')
		tmpdf = tmpdf[['date']].copy()
		start = pd.to_datetime(tmpdf['date'].head(1).values[0]).strftime('%Y-%m-%d')
		end = pd.to_datetime(tmpdf['date'].tail(1).values[0]).strftime('%Y-%m-%d')
		tfstr = get_timeframe_by_df(df.tail(50))
		freq = get_pd_freq_by_tf(tfstr)
		new_idx = pd.date_range(start=start, end=end, freq=freq)
		new_df = pd.DataFrame(index=new_idx).reset_index()
		new_df = new_df.rename(columns={'index': 'date'})
		new_df = merge_df(new_df, df, on='date', how='left')
		rtn = new_df.ffill().dropna().copy()
		sw.end()
		return rtn
	except Exception as e:
		print(
			'ERROR: Error for method normalise_timeseries_df(), ERR - \n%s\n%s' % (
				str(traceback.format_exc()), str(e)
			)
		)
		sys.exit(1)


def get_growth_rate_helper(gr, xop):
	""" A helper for get_grwoth_rate
	"""
	if isinstance(xop, bool) and xop:
		return (abs(gr) * 1) - 1
	else:
		return gr


def get_growth_rate(value_list_df):
	"""Calculates growth rate from a value list dataframe.
		INPUT: Dataframe with columns open, high, low, close, or value.
		OUTPUT: The dataframe structure and columns but with growth rate values.
	"""
	df = value_list_df.copy()
	if df['date'].count() < 4:
		print('WARNING: get_growth_rate() - size is less than 4. Returning False.')
		return False
	df['date'] = pd.to_datetime(df['date'])
	df = df.sort_values(by='date')
	is_ohlc = ('close' in df.columns)
	is_econometric = ('value' in df.columns)
	if is_ohlc:
		df['gr'] = df['close'].pct_change()
		return df[['date', 'gr']].copy()
		df = clean_value_df_cols(df, 'open')
		df = clean_value_df_cols(df, 'high')
		df = clean_value_df_cols(df, 'low')
		df = clean_value_df_cols(df, 'close')
		df['is_pv_neg'] = df['open'] < 0
		df['is_v_pos'] = df['close'] > 0
		df['is_cross_over_c1'] = df['is_pv_neg'] == df['is_v_pos']
		df['is_cross_over'] = df['is_cross_over_c1'] == True
		df['is_cross_over_plus'] = (
			df[df['is_cross_over'] == True]['open'] < df[df['is_cross_over'] == True]['close']
		)
		df['CMinusP'] = df['close'] - df['open']
		df['gr'] = df['CMinusP'] / df['open']
		df['gr'] = df['gr'].astype(float)
		df['gr'] = df['gr'].round(10)
		# When crossing from neg data to pos data, the gr must be abs pos with 1 subtracted
		df['gr'] = df.apply(
			lambda x: get_growth_rate_helper(x['gr'], x['is_cross_over_plus']), axis=1
		)

	elif is_econometric:
		df = clean_value_df_cols(df, 'value')
		df['value'] = df['value'].astype(float)
		df['gr'] = df['value'].pct_change()
		return df[['date', 'gr']].copy()
		df['prev_value'] = df['value'].shift(1)
		df['is_pv_neg'] = df['prev_value'] < 0
		df['is_v_pos'] = df['value'] > 0
		df['is_cross_over_c1'] = df['is_pv_neg'] == df['is_v_pos']
		df['is_cross_over'] = df['is_cross_over_c1'] == True
		df['is_cross_over_plus'] = (
			df[df['is_cross_over'] == True]['prev_value'] < df[df['is_cross_over'] == True]['value']
		)
		df['CMinusP'] = df['value'] - df['prev_value']
		df['gr'] = df['CMinusP'] / df['prev_value']
		# When crossing from neg data to pos data, the gr must be abs pos with 1 subtracted
		df['gr'] = df.apply(
			lambda x: get_growth_rate_helper(x['gr'], x['is_cross_over_plus']), axis=1
		)
	return df[['date', 'gr']].copy()


def get_average_candle_body(window_periods, df):
	"""Calculates the rolling average candle body size which is the abs of open and close.
		INPUT: window_periods, dataframe
		OUTPUT: a dataframe with a new column avg_body_size
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < 4:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	try:
		if is_ohlc:
			value_list_df = clean_value_df_cols(value_list_df, 'open')
			value_list_df = clean_value_df_cols(value_list_df, 'high')
			value_list_df = clean_value_df_cols(value_list_df, 'low')
			value_list_df = clean_value_df_cols(value_list_df, 'close')
			value_list_df['body_size'] = (
				value_list_df['open'] - value_list_df['close']
			).abs()
			value_list_df['avg_body_size'] = value_list_df['body_size'].rolling(
				window_periods
			).mean()
		return value_list_df[['date', 'avg_body_size']].copy()
	except Exception:
		return []


def get_average_candle_tail(window_periods, df):
	"""Calculates the rolling average candle tail size.
		INPUT: window_periods, dataframe
		OUTPUT: a series dataframe with average candle tail sizes
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < 4:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	metric_list = []
	try:
		if is_ohlc:
			value_list_df = clean_value_df_cols(value_list_df, 'open')
			value_list_df = clean_value_df_cols(value_list_df, 'high')
			value_list_df = clean_value_df_cols(value_list_df, 'low')
			value_list_df = clean_value_df_cols(value_list_df, 'close')
			value_list_df['tail_a'] = (
				value_list_df['open'] - value_list_df['low']
			).abs()
			value_list_df['tail_a_avg'] = value_list_df['tail_a'].rolling(
				window_periods
			).mean()
			value_list_df['tail_b'] = (
				value_list_df['close'] - value_list_df['low']
			).abs()
			value_list_df['tail_b_avg'] = value_list_df['tail_b'].rolling(
				window_periods
			).mean()
			value_list_df['is_bullish'] = (
				value_list_df['close'] > value_list_df['open']
			)
			for index, row in value_list_df.iterrows():
				is_bullish = row['is_bullish']
				tail = None
				if is_bullish:
					tail = row['tail_a_avg']
				else:
					tail = row['tail_b_avg']
				metric_list.append(
					{'date': row['date'], 'avg_tail_size': tail}
				)
		return pd.DataFrame(metric_list)
	except Exception:
		return []


def get_average_candle_head(window_periods, df):
	"""Calculates the rolling average candle head size.
		INPUT: window_periods, dataframe
		OUTPUT: a series dataframe with average candle head size
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < 4:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	metric_list = []
	try:
		if is_ohlc:
			value_list_df = clean_value_df_cols(value_list_df, 'open')
			value_list_df = clean_value_df_cols(value_list_df, 'high')
			value_list_df = clean_value_df_cols(value_list_df, 'low')
			value_list_df = clean_value_df_cols(value_list_df, 'close')
			value_list_df['head_a'] = (
				value_list_df['open'] - value_list_df['high']
			).abs()
			value_list_df['head_a_avg'] = value_list_df['head_a'].rolling(
				window_periods
			).mean()
			value_list_df['head_b'] = (
				value_list_df['close'] - value_list_df['high']
			).abs()
			value_list_df['head_b_avg'] = value_list_df['head_b'].rolling(
				window_periods
			).mean()
			value_list_df['is_bullish'] = (
				value_list_df['close'] > value_list_df['open']
			)
			for index, row in value_list_df.iterrows():
				is_bullish = row['is_bullish']
				head = None
				if is_bullish:
					head = row['head_b_avg']
				else:
					head = row['head_a_avg']
				metric_list.append(
					{'date': row['date'], 'avg_head_size': head}
				)
		return pd.DataFrame(metric_list)
	except Exception:
		return []


def get_standard_deviation(window_periods, df):
	"""Calculates the standard deviation of a time series.
		INPUT: window_periods, dataframe
		OUTPUT: a series dataframe of standard deviation values
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < window_periods + 1:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	is_econometric = ('value' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	if is_ohlc:
		value_list_df = clean_value_df_cols(value_list_df, 'close')
		value_list_df['close'] = value_list_df['close'].astype(float)
		value_list_df['std'] = value_list_df['close'].rolling(
			window_periods
		).std()
	elif is_econometric:
		value_list_df = clean_value_df_cols(value_list_df, 'value')
		value_list_df['value'] = value_list_df['value'].astype(float)
		value_list_df['std'] = value_list_df['value'].rolling(
			window_periods
		).std()
	return value_list_df[['date', 'std']].copy()


def get_drawdown_metrics(window_periods, df):
	"""Calulcates drawdown metrics from a time series.
		INPUT: window_periods - one period per detected fall, dataframe
		OUTPUT: python dictionary of drawdown metrics in dataframe series format
				  {avg_duration, max_duration, min_duration, fall_count}
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < window_periods + 1:
		return {
			'avg_duration': pd.DataFrame([]),
			'max_duration': pd.DataFrame([]),
			'min_duration': pd.DataFrame([]),
			'fall_count': pd.DataFrame([])
		}

	is_ohlc = ('close' in value_list_df.columns)
	value_list_df = value_list_df.sort_values(by='date')
	if is_ohlc:
		value_list_df['value'] = value_list_df['close']
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	value_list_df = clean_value_df_cols(value_list_df, 'value')
	value_list_df['previous_value'] = value_list_df['value'].shift(1)
	value_list_df['is_falling'] = (
		value_list_df['value'] < value_list_df['previous_value']
	)
	drawdowns = []
	periods = 0
	exception = 3
	for index, row in value_list_df.iterrows():
		falling = row['is_falling']
		if falling:
			periods = periods + 1
			exception = 0
		else:
			exception = exception + 1
			if exception == 2:
				duration = periods
				periods = 0
				drawdowns.append(
					{
						'date': row['date'],
						'fall': 1,
						'duration': duration
					}
				)
	drawdowns_df = pd.DataFrame(drawdowns)
	drawdowns_df['avg_duration'] = drawdowns_df['duration'].rolling(
		window_periods
	).mean()
	drawdowns_df['max_duration'] = drawdowns_df['duration'].rolling(
		window_periods
	).max()
	drawdowns_df['min_duration'] = drawdowns_df['duration'].rolling(
		window_periods
	).min()
	drawdowns_df['fall_count'] = drawdowns_df['fall'].rolling(
		window_periods
	).sum()
	# print(value_list_df)  # Debug
	# print(drawdowns_df)  # Debug
	drawdowns_df['value'] = drawdowns_df['avg_duration']
	ad_df = drawdowns_df[['date', 'value']].copy()
	drawdowns_df['value'] = drawdowns_df['max_duration']
	md_df = drawdowns_df[['date', 'value']].copy()
	drawdowns_df['value'] = drawdowns_df['min_duration']
	mn_df = drawdowns_df[['date', 'value']].copy()
	drawdowns_df['value'] = drawdowns_df['fall_count']
	fc_df = drawdowns_df[['date', 'value']].copy()
	return {
		'avg_duration': ad_df,
		'max_duration': md_df,
		'min_duration': mn_df,
		'fall_count': fc_df
	}


def get_runup_metrics(window_periods, df):
	"""Calculates runup metrics for a time series.
		INPUT: window_periods - one period per detected runup, dataframe
		OUTPUT: a dictionary of runup metrics
				  {avg_duration, max_duration, min_duration, rise_count}
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < window_periods + 1:
		return {
			'avg_duration': pd.DataFrame([]),
			'max_duration': pd.DataFrame([]),
			'min_duration': pd.DataFrame([]),
			'rise_count': pd.DataFrame([])
		}
	is_ohlc = ('close' in value_list_df.columns)
	value_list_df = value_list_df.sort_values(by='date')
	if is_ohlc:
		value_list_df['value'] = value_list_df['close']
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	value_list_df = clean_value_df_cols(value_list_df, 'value')
	value_list_df['previous_value'] = value_list_df['value'].shift(1)
	value_list_df['is_rising'] = (
		value_list_df['value'] > value_list_df['previous_value']
	)
	runups = []
	periods = 0
	exception = 3
	for index, row in value_list_df.iterrows():
		rising = row['is_rising']
		if rising:
			periods = periods + 1
			exception = 0
		else:
			exception = exception + 1
			if exception == 2:
				duration = periods
				periods = 0
				runups.append(
					{
						'date': row['date'],
						'rise': 1,
						'duration': duration
					}
				)
	runups_df = pd.DataFrame(runups)
	runups_df['avg_duration'] = runups_df['duration'].rolling(
		window_periods
	).mean()
	runups_df['max_duration'] = runups_df['duration'].rolling(
		window_periods
	).max()
	runups_df['min_duration'] = runups_df['duration'].rolling(
		window_periods
	).min()
	runups_df['rise_count'] = runups_df['rise'].rolling(
		window_periods
	).sum()
	# print(value_list_df)  #  Debug
	# print(runups_df)  # Debug
	runups_df['value'] = runups_df['avg_duration']
	ad_df = runups_df[['date', 'value']].copy()
	runups_df['value'] = runups_df['max_duration']
	md_df = runups_df[['date', 'value']].copy()
	runups_df['value'] = runups_df['min_duration']
	mn_df = runups_df[['date', 'value']].copy()
	runups_df['value'] = runups_df['rise_count']
	rc_df = runups_df[['date', 'value']].copy()
	return {
		'avg_duration': ad_df,
		'max_duration': md_df,
		'min_duration': mn_df,
		'rise_count': rc_df
	}


def get_average_true_range(window_periods, df):
	"""Calculates the average true range of an ohlc time series
		INPUT: window_periods, dataframe
		OUTPUT: a dataframe of ATR values
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < (window_periods + 1):
		return []
	is_ohlc = ('close' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	try:
		if is_ohlc:
			value_list_df = clean_value_df_cols(value_list_df, 'open')
			value_list_df = clean_value_df_cols(value_list_df, 'high')
			value_list_df = clean_value_df_cols(value_list_df, 'low')
			value_list_df = clean_value_df_cols(value_list_df, 'close')
			value_list_df['prev_close'] = value_list_df['close'].shift(1)
			value_list_df['a'] = value_list_df['high'] - value_list_df['low']
			value_list_df['b'] = (
				value_list_df['high'] - value_list_df['prev_close']
			).abs()
			value_list_df['c'] = (
				value_list_df['low'] - value_list_df['prev_close']
			).abs()
			value_list_df['true_range'] = value_list_df[
				['a', 'b', 'c']
			].max(axis=1)
			value_list_df['average_true_range'] = value_list_df[
				'true_range'
			].rolling(window_periods).mean()
			# print(value_list_df)  #  Debug
			value_list_df['atr'] = value_list_df['average_true_range']
		return value_list_df[['date', 'atr']].copy()
	except Exception:
		return []


def get_cumsum(df):
	value_list_df = df.copy()
	is_ohlc = ('close' in value_list_df.columns)
	is_econometric = ('value' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	if is_ohlc:
		value_list_df = clean_value_df_cols(value_list_df, 'close')
		value_list_df['value'] = value_list_df['close']
	if is_econometric:
		value_list_df = clean_value_df_cols(value_list_df, 'value')
	value_list_df['cs'] = value_list_df['value'].cumsum()
	return value_list_df[['date', 'cs']].copy()


def get_geometric_return_helper(df_in):
	"""INPUT: Rolling df window returned from pd.DataFrame.rolling
		OUTPUT: The geometric return of the given window
	"""
	df = df_in.copy()
	periodic_sum = df['gr'].sum()
	duration = df['gr'].count()
	# TODO: Fix unittest
	# if duration == 0:
	#   return None
	effective_date = df['date'].tail(1).values[0]
	ar = (abs(periodic_sum)**(1.0 / duration))
	if periodic_sum < 0:
		ar = ar * -1
	row = {'date': effective_date, 'geortn': ar}
	return row


def get_geometric_return(periods_held, df):
	"""Calculates the geometric return for a timeseries so that they can be compared
		with each other.
		INPUT: periods_held - how long you would hold a trade position for each asset comparision
		OUTPUT: a dataframe of returns that can equally be compared with other geometric series
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < periods_held + 1:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	is_econometric = ('value' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	if is_ohlc:
		value_list_df = clean_value_df_cols(value_list_df, 'close')
		value_list_df['value'] = value_list_df['close']
	if is_econometric:
		value_list_df = clean_value_df_cols(value_list_df, 'value')
	gr_df = get_growth_rate(value_list_df[['date', 'value']])
	metric_list = [get_geometric_return_helper(x) for x in gr_df.rolling(periods_held + 1)]
	return pd.DataFrame(metric_list).sort_values(by='date').copy()


def get_max(window_periods, df):
	"""Calculates the rolling max value of a time series.
		INPUT: window_periods, dataframe
		OUTPUT: a dataframe of rolling max values
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < window_periods + 1:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	is_econometric = ('value' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	if is_ohlc:
		value_list_df = clean_value_df_cols(value_list_df, 'close')
	if is_econometric:
		value_list_df = clean_value_df_cols(value_list_df, 'value')
	if is_ohlc:
		value_list_df['max'] = value_list_df['close'].rolling(
			window_periods
		).max()
	else:
		value_list_df['max'] = value_list_df['value'].rolling(
			window_periods
		).max()
	value_list_df['rmax'] = value_list_df['max']
	return value_list_df[['date', 'rmax']].copy()


def get_min(window_periods, df):
	"""Calculates the rolling min value of a time series.
		INPUT: window_periods, dataframe
		OUTPUT: a dataframe of rolling min values
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < window_periods + 1:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	is_econometric = ('value' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	if is_ohlc:
		value_list_df = clean_value_df_cols(value_list_df, 'close')
	if is_econometric:
		value_list_df = clean_value_df_cols(value_list_df, 'value')
	if is_ohlc:
		value_list_df['min'] = value_list_df['close'].rolling(
			window_periods
		).min()
	else:
		value_list_df['min'] = value_list_df['value'].rolling(
			window_periods
		).min()
	value_list_df['rmin'] = value_list_df['min']
	return value_list_df[['date', 'rmin']].copy()


def get_direction_helper(x):
	d = 0
	is_bull = x['is_bull']
	is_bear = x['is_bear']
	if is_bull:
		d = 1
	if is_bear:
		d = -1
	return d


def get_direction(df):
	"""Calculates the positive or negative direction of a time series.
		INPUT: dataframe with gr column
		OUTPUT: a dataframe of directional values - +1 or -1
	"""
	value_list_df = df.copy()
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	value_list_df['is_bull'] = value_list_df['gr'] > 0
	value_list_df['is_bear'] = value_list_df['gr'] < 0
	value_list_df = value_list_df.dropna()
	value_list_df['direction'] = value_list_df.apply(lambda x: get_direction_helper(x), axis=1)
	return value_list_df[['date', 'direction']].copy()


def get_sma(window_periods, df, calc_type='all'):
	"""Calculates simple moving average.
		INPUT: window_periods, dataframe
				 calc_type=all, gains, losses - we can select ema for bear or bull candles only
		OUTPUT: A dataframe of ema values
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < window_periods + 1:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	is_econometric = ('value' in value_list_df.columns)
	if is_ohlc:
		value_list_df = clean_value_df_cols(value_list_df, 'close')
		value_list_df['value'] = value_list_df['close']
	if is_econometric:
		value_list_df = clean_value_df_cols(value_list_df, 'value')
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	value_list_df['previous_value'] = value_list_df['value'].shift(1)
	value_list_df['is_falling'] = (
		value_list_df['value'] < value_list_df['previous_value']
	)
	if calc_type == 'all':
		value_list_df['sma'] = value_list_df['value'].rolling(
			window_periods
		).mean()
	else:
		value_list_df['value'] = value_list_df['close']
		value_list_df['previous_value'] = value_list_df['value'].shift(1)
		value_list_df['is_falling'] = (
			value_list_df['value'] < value_list_df['previous_value']
		)
		value_list_df = value_list_df.sort_values(by='date')
		if calc_type == 'gains':
			value_list_df['mean'] = value_list_df[
				value_list_df['is_falling'] == False
			].rolling(window_periods)['value'].mean()
		elif calc_type == 'losses':
			value_list_df['mean'] = value_list_df[
				value_list_df['is_falling'] == True
			].rolling(window_periods)['value'].mean()
		value_list_df['sma'] = value_list_df['mean']
		value_list_df.ffill(inplace=True)
		# print(value_list_df)  # Debug
	return value_list_df[['date', 'sma']].copy()


def get_ema(window_periods, df, calc_type='all'):
	"""Calculates the exponential moving average. Recent values are weighted higher.
		INPUT: window_periods, dataframe
				 calc_type=all, gains, losses - we can select ema for bear or bull candles only
		OUTPUT: A dataframe of ema values
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < window_periods + 1:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	is_econometric = ('value' in value_list_df.columns)
	if is_ohlc:
		value_list_df = clean_value_df_cols(value_list_df, 'close')
		value_list_df['value'] = value_list_df['close']
	if is_econometric:
		value_list_df = clean_value_df_cols(value_list_df, 'value')
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	value_list_df['previous_value'] = value_list_df['value'].shift(1)
	value_list_df['is_falling'] = (
		value_list_df['value'] < value_list_df['previous_value']
	)
	value_list_df['prev_sma'] = value_list_df['value'].rolling(window_periods).mean()
	k = 2 / (window_periods + 1)
	metric_list = []
	start = True
	count = 0
	ema_prev = 0
	for index, row in value_list_df.iterrows():
		count = count + 1
		is_falling = bool(row['is_falling'])
		skip = True
		if calc_type == 'gains':
			if not is_falling:
				skip = False
		elif calc_type == 'losses':
			if is_falling:
				skip = False
		elif calc_type == 'all':
			skip = False
		if count > (window_periods) + 1 and not skip:
			price_current = row['value']
			if start:
				start = False
				ema_prev = row['prev_sma']
			ema = (price_current * k) + (ema_prev * (1 - k))
			metric_list.append(
				{'date': row['date'], 'ema': ema}
			)
			ema_prev = ema
		else:
			metric_list.append(
				{'date': row['date'], 'ema': ema_prev}
			)
	return pd.DataFrame(metric_list)


def get_relative_strength_index(window_periods, df):
	"""Calculates relative strength index.
		INPUT: window_periods, dataframe
		OUTPUT: a dataframe of RSI values
	"""
	value_list_df = df.copy()
	if value_list_df['date'].count() < window_periods + 1:
		return []
	is_ohlc = ('close' in value_list_df.columns)
	is_econometric = ('value' in value_list_df.columns)
	if is_ohlc:
		value_list_df = clean_value_df_cols(value_list_df, 'close')
		value_list_df['value'] = value_list_df['close']
	if is_econometric:
		value_list_df = clean_value_df_cols(value_list_df, 'value')
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	gains_df = get_ema(window_periods, value_list_df, calc_type='gains')
	gains_df = gains_df.rename(columns={'ema': 'avg_gains'})
	losses_df = get_ema(window_periods, value_list_df, calc_type='losses')
	losses_df = losses_df.rename(columns={'ema': 'avg_losses'})
	value_list_df = merge_df(value_list_df, gains_df, on='date')
	value_list_df = merge_df(value_list_df, losses_df, on='date')
	value_list_df = value_list_df.dropna()
	value_list_df['rsi'] = 100 - (
		100 / (
			1 + (value_list_df['avg_gains'] / value_list_df['avg_losses'])
		)
	)
	return value_list_df[['date', 'rsi']].copy().dropna()


def add_quarterofyear_df(dfp):
	"""Calculates quarter of year and adds it to the input dataframe.
		INPUT: dataframe
		OUTPUT: enriched dataframe with quarter_of_year column
	"""
	df = dfp.copy()
	if 'date' not in df.columns:
		df = df.reset_index()
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df['quarter_of_year'] = pd.to_numeric(df.index.month / 3)
	df['quarter_of_year'] = df['quarter_of_year'].apply(
		lambda x: math.ceil(x)
	)
	return df.reset_index()


def add_monthofyear_df(dfp):
	"""Calculates month of year and adds it to the input dataframe.
		INPUT: dataframe
		OUTPUT: enriched dataframe with month_of_year column
	"""
	df = dfp.copy()
	if 'date' not in df.columns:
		df = df.reset_index()
	df['date'] = pd.to_datetime(df['date'])
	df['month_of_year'] = df['date'].apply(lambda x: x.month)
	return df


def add_dayofweek_df(dfp):
	"""Calculate day of week.
		INPUT: dataframe
		OUTPUT: enriched dataframe with day_of_week column Monday0 to Sunday6
	"""
	df = dfp.copy()
	if 'date' not in df.columns:
		df = df.reset_index()
	df['date'] = pd.to_datetime(df['date'])
	df['day_of_week'] = df['date'].apply(lambda x: x.dayofweek)
	return df


def add_weekofmonth_df(dfp):
	"""Calculate week of month.
		INPUT: dataframe
		OUTPUT: enriched dataframe with week_of_month column
	"""
	df = dfp.copy()
	if 'date' not in df.columns:
		df = df.reset_index()
	df['date'] = pd.to_datetime(df['date'])
	df = df.set_index('date')
	df['week_of_month'] = pd.to_numeric(df.index.day / 7)
	df['week_of_month'] = df['week_of_month'].apply(
		lambda x: math.ceil(x)
	)
	return df.reset_index()


def seasonality_row(value_list_df, time_frame):
	"""A helper method for get_seasonality function.
	"""
	df = value_list_df.copy()
	is_daily = (time_frame == 'daily')
	is_weekly = (time_frame == 'weekly')
	is_monthly = (time_frame == 'monthly')
	is_quarterly = (time_frame == 'quarterly')
	season_key = ''
	if is_daily:
		df = add_dayofweek_df(df)
		season_key = 'day_of_week'
	elif is_weekly:
		df = add_weekofmonth_df(df)
		season_key = 'week_of_month'
	elif is_monthly:
		df = add_monthofyear_df(df)
		season_key = 'month_of_year'
	elif is_quarterly:
		df = add_quarterofyear_df(df)
		season_key = 'quarter_of_year'
	else:
		print('Invalid time frame of ' + str(time_frame))
	df = df.reset_index()
	enriched = []
	for index, row in df.iterrows():
		date = row['date']
		if 'close' in row:
			open_val = row['open']
			high_val = row['high']
			low_val = row['low']
			close_val = row['close']
		elif 'value' in row:
			close_val = row['value']
		season_value = row[season_key]
		if 'close' in row:
			enriched.append(
				{
					'date': date,
					season_key: season_value,
					'open': open_val,
					'high': high_val,
					'low': low_val,
					'close': close_val
				}
			)
		elif 'value' in row:
			enriched.append(
				{
					'date': date,
					season_key: season_value,
					'value': close_val
				}
			)
	return pd.DataFrame(enriched)


def get_seasonality_calcs_tag(x):
	v = x[0]
	insig_pct = x[1]
	if v > insig_pct:
		return 1
	elif v < (abs(insig_pct) * -1):
		return -1
	elif v <= insig_pct and v >= (abs(insig_pct) * -1):
		return 0


def get_seasonality_calcs(
	skeys, growth_rate_df, season_key, seasonality_base_data_point,
	window_periods, insig_pct=None, method='avg'
):
	"""A helper method for get_seasonality function.
	"""
	c = 0
	if 'mondays' in skeys:
		c = -1
	new_skey = ['date']
	for k in skeys:
		if k != 'date':
			c = c + 1
			if method == 'avg':
				growth_rate_df[k] = growth_rate_df[
					growth_rate_df[season_key] == c
				][seasonality_base_data_point].rolling(window_periods).mean()
			elif method == 'count':
				bull_k = '%s_bullc' % (k, )
				bear_k = '%s_bearc' % (k, )
				insig_k = '%s_insigc' % (k, )
				new_skey.extend([bull_k, bear_k, insig_k])
				growth_rate_df[bull_k] = growth_rate_df[(
					growth_rate_df[season_key] == c
				) & (
					growth_rate_df[seasonality_base_data_point] > insig_pct
				)][seasonality_base_data_point].rolling(window_periods).count()
				growth_rate_df[bear_k] = growth_rate_df[(
					growth_rate_df[season_key] == c
				) & (
					growth_rate_df[seasonality_base_data_point] < (abs(insig_pct) * -1)
				)][seasonality_base_data_point].rolling(window_periods).count()
				growth_rate_df[insig_k] = growth_rate_df[(
					growth_rate_df[season_key] == c
				) & (
					growth_rate_df[seasonality_base_data_point] <= insig_pct
				) & (
					growth_rate_df[seasonality_base_data_point] >= (abs(insig_pct) * -1)
				)][seasonality_base_data_point].rolling(window_periods).count()
			elif method == 'tag':
				growth_rate_df[k] = growth_rate_df[
					growth_rate_df[season_key] == c
				][seasonality_base_data_point].apply(
					lambda x: get_seasonality_calcs_tag((x, insig_pct))
				)
			else:
				print('ERROR: No correct method found.')
				sys.exit(1)
	if new_skey.__len__() > 1:
		if method == 'tag':
			# Do not mess up the tagged probability space
			seasonality_df = growth_rate_df[new_skey]
		else:
			seasonality_df = growth_rate_df[new_skey].ffill()
		skeys = new_skey
	else:
		if method == 'tag':
			seasonality_df = growth_rate_df[skeys]
		else:
			seasonality_df = growth_rate_df[skeys].ffill()
	return skeys, seasonality_df


def get_seasonality(window_periods, df, insig_pct=None, method='avg', **kwargs):
	"""Calculates seasonality based on window period growth rate.
		The seasonality value is the rolling average seasonality of that frequency group.
		INPUT: window_periods, dataframe - OHLC or value column data frame
													- all data will be converted to a growth rate
				 insig_pct - Configure this to return a count of bears and bulls
							  - use insig_pct to exclude growth rates that are too small
							  - example - insig_pct=0.05 - 5 percent gr will be counted as too small
		OUTPUT: dataframe with seasonality columns
				  - For normal seasonality returning average growth rate per seasonal period
						** we return a pattern of columns as per below
						- mondays, tuesdays etc. for daily time frames
						- week_1, week_2 etc. for weekly time frames
						- month_1, month_2 etc. for monthly time frames
						- quarter_1, quarter_2 etc. for quarterly time frames
				  - For bear, bull, insig counts for odds calcs per seasonal period
						** we return a pattern of columns as per below
						- mondays_bearc, mondays_bullc, mondays_insigc etc. for daily time frames
						- week_1_bearc, week_2_bearc etc. for weekly time frames
						- month_1_bullc, month_2_bullc etc. for monthly time frames
						- quarter_1_insigc, quarter_2_insigc etc. for quarterly time frames
	"""
	value_list_df = df.copy()
	if value_list_df.empty or value_list_df['date'].count() < 4:
		return pd.DataFrame()
	# is_ohlc = ('close' in value_list_df.columns)
	# is_econometric = ('value' in value_list_df.columns)
	value_list_df['date'] = pd.to_datetime(value_list_df['date'])
	value_list_df = value_list_df.sort_values(by='date')
	if 'gr' not in value_list_df.columns:
		value_list_df = get_growth_rate(value_list_df)
	value_list_df['value'] = value_list_df['gr']
	value_list_df = value_list_df[['date', 'value']].copy()
	date_a = value_list_df['date'].iloc[3]
	date_b = value_list_df['date'].iloc[4]
	date_c = value_list_df['date'].iloc[5]

	time_frame = get_timeframe_by_dates(date_a, date_b, date_c)
	is_daily = (time_frame == 'daily')
	is_weekly = (time_frame == 'weekly')
	is_monthly = (time_frame == 'monthly')
	is_quarterly = (time_frame == 'quarterly')
	# is_annual = (time_frame == 'annual')
	season_key = ''
	if is_daily:
		grouped_seasonality = [
			seasonality_row(g, time_frame) for n, g in value_list_df.set_index(
				'date'
			).groupby(
				pd.Grouper(freq='W')
			)
		]
		season_key = 'day_of_week'
	elif is_weekly:
		grouped_seasonality = [
			seasonality_row(g, time_frame) for n, g in value_list_df.set_index(
				'date'
			).groupby(
				pd.Grouper(freq='Y')
			)
		]
		season_key = 'week_of_month'
	elif is_monthly:
		grouped_seasonality = [
			seasonality_row(g, time_frame) for n, g in value_list_df.set_index(
				'date'
			).groupby(
				pd.Grouper(freq='Y')
			)
		]
		season_key = 'month_of_year'
	elif is_quarterly:
		grouped_seasonality = [
			seasonality_row(g, time_frame) for n, g in value_list_df.set_index(
				'date'
			).groupby(
				pd.Grouper(freq='Y')
			)
		]
		season_key = 'quarter_of_year'
	else:
		print(
			'WARNING: get_seasonality() - Could not detect window time frame - result - ' +
			str(time_frame)
		)
		return []
	metric_list = []
	seasonality_base_data_point = ''
	for group in grouped_seasonality:
		for index, row in group.iterrows():
			metric_list.append(
				{
					'date': row['date'],
					season_key: row[season_key],
					'value': float(row['value'])
				}
			)
	growth_rate_df = pd.DataFrame(metric_list)
	# print(
	#    'growth_rate_df 4 samples: ' + str(growth_rate_df[:4])
	# )
	keys_daily = [
		'date', 'mondays', 'tuesdays', 'wednesdays', 'thursdays', 'fridays', 'saturdays', 'sundays'
	]
	keys_weekly = ['date']
	keys_weekly.extend(['week_%s' % (x, ) for x in range(1, 6)])
	keys_monthly = ['date']
	keys_monthly.extend(['month_%s' % (x, ) for x in range(1, 13)])
	keys_quarterly = ['date']
	keys_quarterly.extend(['quarter_%s' % (x, ) for x in range(1, 5)])
	seasonality_base_data_point = 'value'
	seasonality_option_keys = []
	if is_daily:
		skeys = keys_daily
		skeys, seasonality_df = get_seasonality_calcs(
			skeys, growth_rate_df, season_key, seasonality_base_data_point,
			window_periods, insig_pct=insig_pct, method=method
		)
		seasonality_df = add_dayofweek_df(seasonality_df)
		seasonality_option_keys = ['date', 'day_of_week']
	elif is_weekly:
		skeys = keys_weekly
		skeys, seasonality_df = get_seasonality_calcs(
			skeys, growth_rate_df, season_key, seasonality_base_data_point,
			window_periods, insig_pct=insig_pct, method=method
		)
		seasonality_df = add_weekofmonth_df(seasonality_df)
		seasonality_option_keys = ['date', 'week_of_month']
	elif is_monthly:
		skeys = keys_monthly
		skeys, seasonality_df = get_seasonality_calcs(
			skeys, growth_rate_df, season_key, seasonality_base_data_point,
			window_periods, insig_pct=insig_pct, method=method
		)
		seasonality_df = add_monthofyear_df(seasonality_df)
		seasonality_option_keys = ['date', 'month_of_year']
	elif is_quarterly:
		skeys = keys_quarterly
		skeys, seasonality_df = get_seasonality_calcs(
			skeys, growth_rate_df, season_key, seasonality_base_data_point,
			window_periods, insig_pct=insig_pct, method=method
		)
		seasonality_df = add_quarterofyear_df(seasonality_df)
		seasonality_option_keys = ['date', 'quarter_of_year']
	for row in list(kwargs.keys()):
		if kwargs[row] is True:
			if row not in seasonality_option_keys:
				seasonality_option_keys.append(row)
	if seasonality_option_keys.__len__() == 2:
		for k in skeys:
			if k not in seasonality_option_keys:
				seasonality_option_keys.append(k)
	df = seasonality_df[seasonality_option_keys].reset_index().copy()
	del df['index']
	return df


def straighten_daily(df_in):
	"""Converts a daily dataframe that may be skewed, into a standard daily series.
		INPUT: dataframe in apparent daily ohlc format
		OUTPUT: dataframe in daily ohlc format
	"""
	df = df_in.copy()
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
	df = df.sort_values(by='date')
	mind = str(df.head(1)['date'].values[0])
	maxd = str(df.tail(1)['date'].values[0])
	tf = pd.DataFrame()
	tf["date"] = pd.date_range(mind, maxd, freq="D")
	tf['date'] = pd.to_datetime(tf['date'])
	tf = merge_df(tf, df, na='ffill', how='left', on='date')
	df = tf
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
	return df.copy()


def straighten_monthly(df_in):
	"""Converts a monthly dataframe that may be skewed, into a standard monthly series.
		INPUT: dataframe in apparent monthly ohlc format
		OUTPUT: dataframe in monthly ohlc format
	"""
	df = df_in.copy()
	df['date'] = pd.to_datetime(df['date'])
	df = df.sort_values(by='date')
	mind = str(df.head(1)['date'].values[0])
	maxd = str(df.tail(1)['date'].values[0])
	tf = pd.DataFrame()
	tf["date"] = pd.date_range(mind, maxd, freq="M")
	tf['date'] = pd.to_datetime(tf['date'])
	tf = merge_df(tf, df, na='dropna', how='left', on='date')
	df = tf
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
	return df.copy()


def straighten_quarterly(df_in):
	"""Converts a quarterly dataframe that may be skewed, into a standard quarterly series.
		INPUT: dataframe in apparent quarterly ohlc format
		OUTPUT: dataframe in quarterly ohlc format
	"""
	df = df_in.copy()
	df['date'] = pd.to_datetime(df['date'])
	df = df.sort_values(by='date')
	mind = str(df.head(1)['date'].values[0])
	maxd = str(df.tail(1)['date'].values[0])
	tf = pd.DataFrame()
	tf["date"] = pd.date_range(mind, maxd, freq="Q")
	tf['date'] = pd.to_datetime(tf['date'])
	tf = merge_df(tf, df, na='dropna', how='left', on='date')
	df = tf
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
	return df.copy()


def convert_daily_to_weekly(df_in):
	"""Converts a daily dataframe into an end of week ohlc series.
		INPUT: dataframe in daily ohlc format
		OUTPUT: dataframe in weekly ohlc format
	"""
	df = df_in.copy()
	candle_cols = ['open', 'high', 'low', 'close']
	c_candle = 0
	for col in df.columns:
		if col in candle_cols:
			c_candle = c_candle + 1
	is_candle = False
	if c_candle == 4:
		is_candle = True
	df['date'] = pd.to_datetime(df['date'])
	df = df.sort_values(by='date')
	grouped_seasonality = [
		seasonality_row(g, 'daily') for n, g in df.set_index(
			'date'
		).groupby(
			pd.Grouper(freq='W')
		)
	]
	weekly = []
	prev_date = None
	for week in grouped_seasonality:
		if 'date' not in week:
			continue
		curr_date = pd.to_datetime(week['date'].tail(1).values[0])
		if prev_date:
			diff = abs(curr_date - prev_date)
			if diff < timedelta(days=7):
				continue
		if is_candle:
			weekly.append({
				'date': curr_date,
				'open': week['open'].head(1).values[0],
				'high': week['high'].max(),
				'low': week['low'].min(),
				'close': week['close'].tail(1).values[0]
			})
		else:
			weekly.append({
				'date': curr_date,
				'value': week['value'].tail(1).values[0]
			})
		prev_date = curr_date
	return pd.DataFrame(weekly)


def convert_ohlc_daily_to_weekly(df_in):
	return convert_daily_to_weekly(df_in)


def convert_daily_to_monthly(df_in):
	"""Converts a daily dataframe into an end of month ohlc series.
		INPUT: dataframe in daily ohlc format
		OUTPUT: dataframe in monthly ohlc format
	"""
	return straighten_monthly(df_in)


def convert_ohlc_daily_to_monthly(df_in):
	return convert_daily_to_monthly(df_in)


def convert_daily_to_quarterly(df_in):
	"""Converts a daily dataframe into an end of quarter ohlc series.
		INPUT: dataframe in daily ohlc format
		OUTPUT: dataframe in quarterly ohlc format
	"""
	return straighten_quarterly(df_in)


def convert_ohlc_daily_to_quarterly(df_in):
	return convert_daily_to_quarterly(df_in)


def convert_weekly_to_monthly(df_in):
	""" We convert weekly to daily. Then find the end of month.
	"""
	df = df_in.copy()
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
	df = df.sort_values(by='date')
	mind = str(df.head(1)['date'].values[0])
	maxd = str(df.tail(1)['date'].values[0])
	tf = pd.DataFrame()
	tf["date"] = pd.date_range(mind, maxd, freq="D")
	tf['date'] = pd.to_datetime(tf['date'])
	tf = merge_df(tf, df, how='outer', on='date')
	return convert_daily_to_monthly(tf)


def convert_weekly_to_quarterly(df_in):
	""" We convert weekly to daily. Then find the end of quarter.
	"""
	df = df_in.copy()
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
	df = df.sort_values(by='date')
	mind = str(df.head(1)['date'].values[0])
	maxd = str(df.tail(1)['date'].values[0])
	tf = pd.DataFrame()
	tf["date"] = pd.date_range(mind, maxd, freq="D")
	tf['date'] = pd.to_datetime(tf['date'])
	tf = merge_df(tf, df, how='outer', on='date')
	return convert_daily_to_quarterly(tf)


def convert_monthly_to_quarterly(df_in):
	""" We convert weekly to daily. Then find the end of quarter.
	"""
	df = df_in.copy()
	df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
	df = df.sort_values(by='date')
	mind = str(df.head(1)['date'].values[0])
	maxd = str(df.tail(1)['date'].values[0])
	tf = pd.DataFrame()
	tf["date"] = pd.date_range(mind, maxd, freq="D")
	tf['date'] = pd.to_datetime(tf['date'])
	tf = merge_df(tf, df, na='ffill', how='outer', on='date')
	return convert_daily_to_quarterly(tf)


def add_array_to_df(column, col_name, df):
	""" Takes a raw input array and adds it as a column to the reference pandas series.
		 Shifts the array to the right/head.

		INPUT: column - list - array of any size - singl value
				 col_name - string - the name of the new dataframe column
				 df - DataFrame - any time series pandas DataFrame
		OUTPUT: the df DataFrame but with an added column
	"""
	column = list(column)
	df_size = df[0].count()
	col_size = column.__len__()
	if col_size > df_size:
		raise ValueError('The dataframe is too small to have the column added')
	fill_size = df_size - column.__len__()
	new_column = [None for x in range(0, fill_size)]
	new_column.extend(column)
	df[col_name] = new_column
	return df.copy()


def convert_array_to_df_by_ref(array, col_name, reference, delete_cols=[]):
	""" Uses an existing df and adds the array by shifting values to the right.
		 Deletes any specified columns.
	"""
	df = add_array_to_df(array, col_name, reference)
	for col in delete_cols:
		del df[col]
	return df.copy()


def scrub_csv_flatfile(dump_location, sed_pattern=None):
	"""!! DO NOT USE. Scrubbing via persistent storage locks the os disk writes.
		!! Load into memory once, then scrub, filter, and transform from there.
		Uses unix sed to scrub the flatfile and remove unwanted characters before consuming.
		INPUT: dump_location - absolute or wildcard
				 sed_pattern - overrides the default module.SED_FILTER constant value
		OUTPUT: returns the /tmp file location of the cleaned flatfile
	"""
	try:
		return dump_location

	except Exception as e:
		print(
			'ERROR: Exception for scrub_csv_flatfile() ERR - \n' + str(traceback.format_exc()) +
			'\n' + str(e)
		)
		sys.exit(1)


def flatfile_to_df(dump_location, **kwargs):
	"""Import into pandas with auto encoding detection and flat file scrubbing.
		If auto codec detection fails, we write to a new file as utf-8 and then import.
		INPUT: filesystem path location of zip, csv, xls, xlsx
		OUTPUT: array of dataframes
	"""
	# Get the filename extention
	file_name_ext = 'csv'
	if isinstance(dump_location, str):
		file_name_ext = dump_location.split('/')[-1].split('.')[-1]
	elif isinstance(dump_location, list):
		file_name_ext = dump_location[0].split('/')[-1].split('.')[-1]
	else:
		file_name_ext = 'csv'

	# Get all absolute locations via wildcard evaluation
	dump_location_arr = []
	if '*' in dump_location:
		for loc in rlcom.filesys().ls_wildcard(dump_location):
			dump_location_arr.append(loc)
		if dump_location_arr.__len__() == 0:
			return []

	if isinstance(dump_location, list):
		dump_location_arr = dump_location

	if file_name_ext == 'zip':
		#
		# Zip archived csv flatfiles. We convert the csv to utf before returning.
		if 'sep' not in kwargs:
			kwargs.update({'sep': ','})
		tmp_zip_dir = '/tmp/tmpzip-' + str(rlcom.rndgen_number(1, 1000))  # nosec
		rlcom.filesys().make_dir(tmp_zip_dir)
		dump_location_abs = rlcom.filesys().ls_wildcard(dump_location)[0]
		rlcom.filesys().unzip(dump_location_abs, tmp_zip_dir)
		file_arr = rlcom.filesys().ls_wildcard(tmp_zip_dir + '/*.csv')
		df_arr = []
		for dumpf in file_arr:
			if dumpf == '':
				continue
			tmp_csv = dumpf
			try:
				df_arr.append(
					pd.read_csv(
						tmp_csv, encoding=rlcom.filesys().chardetect(tmp_csv), **kwargs
					)
				)
			except Exception:
				if not rlcom.filesys().is_utf(tmp_csv):
					rlcom.filesys().save_utf(tmp_csv, tmp_csv + '.utf')
					tmp_csv = tmp_csv + '.utf'
				df_arr.append(
					pd.read_csv(tmp_csv, **kwargs)
				)
				time.sleep(10)
				if '.utf' in tmp_csv:
					rlcom.filesys().remove(tmp_csv)
		rlcom.filesys().remove(tmp_zip_dir)
		return df_arr

	elif file_name_ext == 'csv' and dump_location_arr.__len__() == 0:
		#
		# CSV flatfile. No location arr from wildcard ls evaluation. One source only.
		if 'sep' not in kwargs:
			kwargs.update({'sep': ','})
		tmp_csv = dump_location
		df_arr = []
		try:
			df_arr.append(
				pd.read_csv(
					tmp_csv, encoding=rlcom.filesys().chardetect(tmp_csv), **kwargs
				)
			)
		except Exception:
			if not rlcom.filesys().is_utf(tmp_csv):
				tmp_utf = '/tmp/tmpf-' + str(rlcom.rndgen_number(1, 1000)) + '.utf'  # nosec
				rlcom.filesys().save_utf(tmp_csv, tmp_utf)
				tmp_csv = tmp_utf
			df_arr = [pd.read_csv(tmp_csv, **kwargs)]
			if '.utf' in tmp_csv:
				rlcom.filesys().remove(tmp_csv)
		return df_arr

	elif file_name_ext == 'csv' and dump_location_arr.__len__() > 0:
		#
		# CSV flatfiles. Many sources from ls wildcard evaluation.
		if 'sep' not in kwargs:
			kwargs.update({'sep': ','})
		df_arr = []
		for d in dump_location_arr:
			tmp_csv = d
			try:
				df_arr.append(
					pd.read_csv(
						tmp_csv, encoding=rlcom.filesys().chardetect(tmp_csv), **kwargs
					)
				)
			except Exception:
				if not rlcom.filesys().is_utf(tmp_csv):
					tmp_utf = '/tmp/tmpf-' + str(rlcom.rndgen_number(1, 1000)) + '.utf'  # nosec
					rlcom.filesys().save_utf(tmp_csv, tmp_utf)
					tmp_csv = tmp_utf
				df_arr.append(pd.read_csv(tmp_csv, **kwargs))
				if '.utf' in tmp_csv:
					rlcom.filesys().remove(tmp_csv)
		return df_arr

	elif file_name_ext == 'xls' or file_name_ext == 'xlsx':
		#
		# Excel flatfile. Requires optional pandas dependency xlrd pypi package.
		engine = ''
		if file_name_ext == 'xls':
			engine = 'xlrd'
		elif file_name_ext == 'xlsx':
			engine = 'openpyxl'
		else:
			raise TypeError('ERROR: flatfile_to_df() - File extension not supported.')
		if dump_location_arr.__len__() == 0:
			return [pd.read_excel(dump_location, engine=engine, **kwargs)]
		else:
			df_arr = []
			for d in dump_location_arr:
				df_arr.append(pd.read_excel(d, engine=engine, **kwargs))
			return df_arr


# ================================================================================================
# Data Pre Processing Abstractions
# ================================================================================================


class Normalise():
	def to_log(self, x, ev=None, dec=8):
		"""Return log with x as either, float, int, fraction, or DataFrame type.
			Error handles attempts at converting a negative to a log.
		"""
		try:
			if isinstance(x, float) or isinstance(x, int):
				if isinstance(x, math.nan):
					raise ValueError(
						'ERROR: Normalise() - to_log() could not convert type x of %s with value %s' % (
							str(type(x)), str(x)
						)
					)
				if x <= 0:
					raise ValueError(
						'ERROR: Normalise() - to_log() No Negatives - type x of %s with value %s' % (
							str(type(x)), str(x)
						)
					)
				x = float(x)
				if not ev:
					return round(math.log(x), dec)
				else:
					return round(math.log(x, ev), dec)
			elif isinstance(x, pd.core.frame.DataFrame):
				x['is_neg'] = x < 0
				neg_count = x[x['is_neg'] == True].count()
				if neg_count != 0:
					raise ValueError(
						'ERROR: Normalise() - to_log() No Negatives - type x of %s with value \n%s' % (
							str(type(x)), str(x)
						)
					)
				return x.apply(lambda x: math.log(x))
			elif isinstance(x, F):
				if x == 0:
					x = 0.01
				return round(float(math.log(float(x))), dec)
			else:
				raise ValueError(
					'ERROR: Normalise() - to_log() could not convert type x of %s' % (str(type(x)),)
				)

		except Exception as e:
			print('DEBUG: x=%s, type(x)=%s' % (x, type(x)))
			raise Exception(
				'ERROR: Exception for Normalise().to_log() ERR - \n' + str(traceback.format_exc()) +
				'\n' + str(e)
			)

	def to_fraction(self, x):
		return F().from_float(x)

	def to_metric(self, x):
		n = x.numerator
		d = x.denominator
		return n / d

	def odds_to_fav_prop(self, x):
		f = x
		if isinstance(f, float):
			f = self.to_fraction(x)
		n = f.numerator
		d = f.denominator
		world = n + d
		prop = F(n, world)
		return prop

	def odds_to_unfav_prop(self, x):
		f = x
		if isinstance(f, float):
			f = self.to_fraction(x)
		n = f.numerator
		d = f.denominator
		world = n + d
		uf = world - n
		prop = F(uf, world)
		return prop

	def prop_to_fav_odds(self, x):
		f = x
		if isinstance(f, float):
			f = self.to_fraction(x)
		fav = f.numerator
		world = f.denominator
		if world == fav:
			# return an upper limit of 1000:1 if we are 100% correct in favour within the world space
			return F(100, 1)
		ufav = world - fav
		odds = F(fav, ufav)
		return odds

	def prop_to_unfav_odds(self, x):
		f = x
		if isinstance(f, float):
			f = self.to_fraction(x)
		fav = f.numerator
		world = f.denominator
		if world == fav:
			# return an upper limit of 1000:1 if we are 100% unfavourable within the world space
			return F(100, 1)
		ufav = world - fav
		odds = F(ufav, fav)
		return odds

	def normalise_set_to_one(self, arr):
		""" Scale a list to always have a sum of 1.
		"""
		if isinstance(arr, list) and arr.__len__() == 0:
			print('ERROR: normalise_set_to_one() - arr input length cannot be zero.')
			sys.exit(1)
		if type(arr[0]) == F:
			arr_new = []
			for f in arr:
				arr_new.append(float(f))
			arr = arr_new
		s = [arr]
		data_scaler = preprocessing.Normalizer(norm='l1').fit(s)
		data_rescaled = data_scaler.transform(s)
		rtn = []
		for w in data_rescaled[0]:
			rtn.append(F(w).limit_denominator(100))
		return rtn

	def scale_set(self, arr, mini, maxi):
		""" Scale a list between a range.
		"""
		s = arr
		if isinstance(s, list):
			s = pd.DataFrame(s).values
		elif isinstance(s, pd.core.frame.DataFrame):
			s = s.values
		data_scaler = preprocessing.MinMaxScaler(feature_range=(mini, maxi))
		data_rescaled = data_scaler.fit_transform(s)
		return data_rescaled[0]

	def reset_cumsum(self, df):
		""" Return a data frame of cumsum value with the first record always being 0.
			 INPUT: key value dataframe that is a cumsum
			 OUTPUT: key value dataframe that is a cumsum with first record as zero
						the value column is named: norm_cumsum
		"""
		df = df.sort_values(by='date').copy()
		if df.columns.__len__() != 2:
			raise ValueError('ERROR: Dataframe must be key value pair with only two columns')
		vcol = df.columns[1]
		first_record = df.head(1)[vcol].values[0]
		df['norm_cumsum'] = df[vcol] - first_record
		return df[[df.columns[0], vcol]].copy()


# ================================================================================================
# Multi Criteria Decision Making Abstractions
# ================================================================================================
class MCDM():
	def wsm(self, values, weights):
		""" Weighted Sum Model
			 INPUT  : values  - Array of values to blend
						 weights - Array of weights all equal to 1 when summed
			 OUTPUT : Single averaged value
		"""
		wsm = 0
		Ws = 0  # Sum of all the weights
		for w in weights:
			Ws = Ws + w
		if Ws != 1:
			print('DEBUG: values=%s, weights=%s' % (values, weights))
			raise ValueError('ERROR: wsm() - Sum of all weights must equal 1')
		if values.__len__() != weights.__len__():
			print('DEBUG: values=%s, weights=%s' % (values, weights))
			raise ValueError('ERROR: wsm() - values and weights must have the same array size')
		if values.__len__() == 0 or weights.__len__() == 0:
			print('DEBUG: values=%s, weights=%s' % (values, weights))
			raise ValueError('ERROR: wpm() - values and weights must not be empty arrays')
		for v, w in zip(values, weights):
			wsm = wsm + (v * w)
		return wsm

	def wpm(self, values, weights):
		""" Weighted Product Model
			 INPUT  : values  - Array of values to blend
						 weights - Array of weights all equal to 1 when summed
			 OUTPUT : Single averaged value
		"""
		wpm = 0
		Ws = 0  # Sum of all the weights
		for w in weights:
			Ws = Ws + w
		if Ws != 1:
			raise ValueError('ERROR: wpm() - Sum of all weights must equal 1')
		if values.__len__() != weights.__len__():
			raise ValueError('ERROR: wpm() - values and weights must have the same array size')
		if values.__len__() == 0 or weights.__len__() == 0:
			raise ValueError('ERROR: wpm() - values and weights must not be empty arrays')
		for v, w in zip(values, weights):
			wpm_i = (v**w)
			if wpm == 0:
				wpm = wpm_i
			else:
				wpm = wpm * wpm_i
		return wpm

	def max(self, dict_arr):
		""" Given a dictionary of values, find the largest.
			 INPUT  : A dictionary of bayesian matrix evidence. A single evidence row.
			 OUTPUT : The dictionary of the evidence given MAX hypotheses
		"""
		pass

	def feature_target_forecast(self, features_df, target_df, window):
		""" Wrap ML and AI feature target methods and performance rank them for best forecasted fit.
			 Non linear regression methods only. UIE is probability space forward looking, classifier.
				- decision trees, svm, neural nets
				- maybe UIE can be used to performance benchmark target anticipation performance
					out of sample?
			Low priority. Left here for idea storage.
		"""
		pass


# ================================================================================================
# Exploratory Financial DataScience Preformatted Graphs - Line & Bar
# Scatter for pattern recognition. Box and wiskers for seasonality.
# ================================================================================================
class mp():
	def __set_daily_format(self, ax):
		mondays = WeekdayLocator(MONDAY)
		alldays = DayLocator()
		weekFormatter = DateFormatter('Monday, %b %d, %Y')
		dayFormatter = DateFormatter('%d')
		ax.xaxis.set_major_locator(mondays)
		ax.xaxis.set_minor_locator(alldays)
		ax.xaxis.set_major_formatter(weekFormatter)
		ax.xaxis.set_minor_formatter(dayFormatter)

	def line_chart(self, df, **kwargs):
		""" Preformatted line chart with dataframe date reindexed.
		"""
		dft2 = df.copy()
		dft2['date'] = date2num(dft2['date'])
		dft2 = dft2.set_index('date')
		fig, ax = plt.subplots()
		fig.subplots_adjust(bottom=0.2)
		dft2.plot.line(ax=ax, **kwargs)
		mondays = WeekdayLocator(MONDAY)
		alldays = DayLocator()
		weekFormatter = DateFormatter('Monday, %b %d, %Y')
		dayFormatter = DateFormatter('%d')
		ax.xaxis.set_major_locator(mondays)
		ax.xaxis.set_minor_locator(alldays)
		ax.xaxis.set_major_formatter(weekFormatter)
		ax.xaxis.set_minor_formatter(dayFormatter)
		ax.xaxis_date()
		ax.grid(True)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.gca().set_yticklabels(ax.get_yticks())
		y_formatter = ticker.ScalarFormatter(useOffset=False)
		ax.yaxis.set_major_formatter(y_formatter)
		ax.autoscale_view()
		plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
		fig.set_size_inches(18.5, 10.5)
		return plt

	def bar_chart(self, df, **kwargs):
		""" Preformatted bar chart with dataframe date reindexed.
		"""
		dft2 = df.copy()
		# dft2['date'] = pd.to_datetime(dft2['date'], format='%Y-%m-%d')
		dft2 = dft2.set_index('date')
		fig, ax = plt.subplots()
		fig.subplots_adjust(bottom=0.2)
		dft2.plot.bar(ax=ax, **kwargs)
		# mondays = WeekdayLocator(MONDAY)
		alldays = DayLocator()
		# weekFormatter = DateFormatter('Monday, %b %d')
		dayFormatter = DateFormatter('%b %d, %Y')
		# ax.xaxis.set_major_locator(mondays)
		ax.xaxis.set_minor_locator(alldays)
		# ax.xaxis.set_major_formatter(weekFormatter)
		ax.xaxis.set_minor_formatter(dayFormatter)
		ax.xaxis_date()
		ax.grid(True)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.gca().set_yticklabels(ax.get_yticks())
		y_formatter = ticker.ScalarFormatter(useOffset=False)
		ax.yaxis.set_major_formatter(y_formatter)
		ax.autoscale_view()
		plt.setp(plt.gca().get_xticklabels(which='both'), rotation=45, horizontalalignment='right')
		fig.set_size_inches(18.5, 10.5)
		return plt

	def scatter_chart(self, dft2, **kwargs):
		""" Preformatted scatter chart.
		"""
		fig, ax = plt.subplots()
		fig.subplots_adjust(bottom=0.2)
		dft2.plot.scatter(ax=ax, **kwargs)
		ax.grid(True)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.gca().set_yticklabels(ax.get_yticks())
		y_formatter = ticker.ScalarFormatter(useOffset=False)
		ax.yaxis.set_major_formatter(y_formatter)
		ax.autoscale_view()
		plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='center')
		fig.set_size_inches(18.5, 10.5)
		return plt
