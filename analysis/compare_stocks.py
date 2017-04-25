import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Plot beta and alpha between two stocks
def plot_alpha_beta(df, beta_y, alpha_y, show_plot=True, save_plot=False):
	s1 = df.columns[0]
	s2 = df.columns[1]
	rets1 = df[s1]
	df.plot(kind='scatter', x=s1, y=s2)
	plt.plot(rets1, beta_y*rets1 + alpha_y, '-', color='r')
	plt.xlabel('Daily Returns of ' + str(s1))
	plt.ylabel('Daily Returns of ' + str(s2))
	Title = 'Beta  : ' + str(beta_y) + ' ' + 'Alpha : ' + str(alpha_y) + ' ' + 'Correlation : ' + str(df.corr(method='pearson').iat[0,1])
	plt.title(Title)

	if show_plot:
		plt.show()
	if save_plot:
		plt.savefig('Alpha-Beta : ' + str(s2) + ' ~ ' + str(s1))


def get_alpha_beta(rets1, rets2):
	beta_y, alpha_y = np.polyfit(rets1, rets2, 1)
	return round(beta_y,2), round(alpha_y,2)


def get_correlation_between_returns(rets1, rets2):
	df = pd.concat([rets1, rets2], axis = 1)
	return df.corr(method='pearson').iat[0,1]

#Use Index as Nifty50 by default
def save_beta_alpha(df, use_index='Nifty50'):

	symbols = list(df.columns)

	ABDF = pd.DataFrame(index=symbols, columns=['Beta', 'Alpha'])

	market_rets = df[use_index].pct_change(1)
	market_rets = market_rets.fillna(0)

	for symbol in symbols:
		#Ignore Market Indices
		if symbol != 'Nifty50' and symbol != 'Nifty500':

			stock_rets = df[symbol].pct_change(1)
			stock_rets = stock_rets.fillna(0)
			
			beta, alpha = get_alpha_beta(market_rets, stock_rets)
			correlation = get_correlation_between_returns(market_rets, stock_rets)

			ABDF.loc[symbol] = pd.Series({'Beta':beta, 'Alpha':alpha})

	ABDF.to_csv('12/alpha_beta.csv')
	return ABDF


#Get global statistics about stocks
#Mean, Median, Std
def global_stats(df):

	#Global Statistics
	mean   = df.mean()
	median = df.median()
	std    = df.std()

	global_df = pd.concat([mean, median, std], axis=1)
	global_df.columns = ['Mean', 'Median', 'Std']

	#Save the Global Dataframe
	global_df.to_csv('12/Global_Statistics.csv')

	print global_df

	return global_df

#Histograms
def Histograms(df, symbol1, symbol2, show_plot=True, save_plot=False):
	rets1 = df[symbol1].pct_change(1)
	rets2 = df[symbol2].pct_change(1)
	rets1.hist(bins=20, label=symbol1)
	rets2.hist(bins=20, label=symbol2)

	#plt.axvline(rets.mean(), color='w', linestyle='dashed', linewidth=2)
	#plt.axvline(rets.std(), color='r', linestyle='dashed', linewidth=2)
	#plt.axvline(-rets.std(), color='r', linestyle='dashed', linewidth=2)

	plt.title('Compare Returns')
	plt.xlabel('Daily_Returns')
	plt.legend(loc='best')

	if show_plot:
		plt.show()
		plt.close()

	if save_plot:
		plt.savefig('Price Histogram of ' + str(symbol1) + ' & ' + str(symbol2) + '.png')

	print 'Kurtosis of daily_returns of ' + symbol1 + '\t' + str(rets1.kurtosis())
	print 'Kurtosis of daily_returns of ' + symbol2 + '\t' + str(rets2.kurtosis())


#Save a df of all stocks close for a time range
#Club the Nifty50 and Nifty500 index also
def prices(file_list, start, end, nifty50, nifty500, save_df=False):

	df = pd.DataFrame()

	for stock_file in file_list:
		symbol = stock_file.split('/')[-1][:-4]
		
		stock = pd.read_csv(stock_file)
		stock = stock.set_index('Date')
		close = stock.loc[start:end]['Close']

		df = pd.concat([df, close], axis=1)
		df = df.rename(columns={'Close':symbol})

	#Concat Nifty50
	nifty50 = nifty50.set_index('Date')
	nifty50_close = nifty50.loc[start:end]['Close']
	df = pd.concat([df, nifty50_close], axis=1)
	df = df.rename(columns={'Close':'Nifty50'})

	#Concat Nifty500
	nifty500 = nifty500.set_index('Date')
	nifty500_close = nifty500.loc[start:end]['Close']
	df = pd.concat([df, nifty500_close], axis=1)
	df = df.rename(columns={'Close':'Nifty500'})
	df.index.name = 'Date'

	#Fill NA - Forward & Backward
	df.fillna(method='ffill', inplace=True)
	df.fillna(method='bfill', inplace=True)

	#print df

	#Save the Dataframe
	if save_df:
		df.to_csv('Compare_Stocks_Prices.csv')

	return df
