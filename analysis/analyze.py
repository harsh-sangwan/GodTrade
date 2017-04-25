import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import numpy as np
import talib
from datetime import datetime, timedelta
import os


import Portfolio
import compare_stocks
import trading_strategies
import market_tracker

if __name__ == "__main__":

	now = datetime.now()
	start_date = '2016-12-10'
	#end_date   = str(now.year)+'-'+str(now.month)+'-'+str(now.day)
	end_date   = '2017-02-1'


	cap_path = '../data/NSE/Cap-Size/'
	cap_stocks = []
	symbols = []

	#Nifty50 data
	nifty50 = pd.read_csv('../data/NSE/Indices/^NSEI.csv')

	#Nifty500 data
	nifty500 = pd.read_csv('../data/NSE/Indices/^CRSLDX.csv')
	#print len(os.listdir(cap_path))

	for filename in os.listdir(cap_path):
		if filename != '.DS_Store':

			symbol = filename[:-4]
			symbols.append(str(symbol))

			df = pd.read_csv(cap_path+str(filename))
			#print df
			df = df.set_index('Date')

			close = df.loc[start_date:end_date]['Close']

			cap_stocks.append(cap_path+str(filename))


	#rvdf = market_tracker.returns_vs_volatility(cap_stocks, symbols, end_date, 35)

	#market_tracker.top_performers(rvdf, 15)
	#market_tracker.worst_performers(rvdf, 15)

	Prices_DF 		= compare_stocks.prices(cap_stocks, start_date, end_date, nifty50, nifty500, save_df=True)

	Alpha_Beta_DF   = compare_stocks.save_beta_alpha(Prices_DF, use_index='Nifty500')

	#Global_Stats_DF = compare_stocks.global_stats(Prices_DF)


	BollingerTS = trading_strategies.Bollinger(MA_window=20, mean_k_sigma=2.0, buy_k_sigma=0.5, sell_k_sigma=0.5)
	trading_strategies.Bollinger.Trading_Strategy(BollingerTS, Prices_DF, Alpha_Beta_DF)


	#compare_stocks.Histograms(Prices_DF, 'MINDTREE.NS', 'SUPREMEIND.NS')

	#test_symbols = ['MINDTREE.NS', 'PETRONET.NS', 'SUPREMEIND.NS', 'MERCK.NS', 'TRENT.NS']
	#Portfolio.Optimization(Prices_DF, test_symbols)

	#If want for last n days, give a number after end_date, else -1
	'''result_files = []
	for i in range(0,3) :
		end_date = datetime.now() - i*timedelta(weeks=1)
		end_date = end_date.strftime('%Y-%m-%d')
		result_files.append('09/Results_'+str(end_date)+'.csv')
		highest_return(cap_stocks, symbols, end_date, 5, '11/Results_'+str(end_date)+'.csv')'''

	#Compare Rankings
	#for i in range(1, 2):
	#compare_rankings(result_files[1],result_files[0], symbols)'''