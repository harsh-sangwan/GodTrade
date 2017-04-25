import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def plotting_time_series(dates, y, symbol, ylabel):
	'''df.reset_index(inplace=True)
	df['Date'] = pd.to_datetime(df['Date'])
	df = df.set_index('Date')
	df['Symbol'].plot()'''
	dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
	fig = plt.figure(figsize=(18, 18))
	plt.plot_date(dates, y, 'b-')
	plt.grid()
	plt.xlabel('Dates')
	plt.ylabel(ylabel)
	plt.title(symbol)	
	return plt

	