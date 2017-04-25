import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotting_time_series import plotting_time_series
import pybacktest

class Bollinger(object):

	def __init__(self, MA_window=20, mean_k_sigma=2.0, buy_k_sigma=0.5, sell_k_sigma=0.5):
		self.MA_window = MA_window

		self.mean_k_sigma = mean_k_sigma
		self.buy_k_sigma  = buy_k_sigma
		self.sell_k_sigma = sell_k_sigma

	def bollinger_bands(self, rmean, rstd, factor):
		upper = rmean + factor * rstd
		lower = rmean - factor * rstd
		return upper, lower

	def plot_bollinger_bands(self, df, symbol, roll_mean, upper, lower, buy_upper, buy_lower, beta, alpha, buy_sell_signal, show_plot=True, save_plot=True):
		
		dates = df.index.tolist()

		plt = plotting_time_series(dates, df[symbol], symbol, 'Closing_Prices')

		plt.plot(roll_mean, 'g-', label='SMA' + str(self.MA_window))
		plt.plot(upper, 'r-', label='SMA' + str(self.MA_window) + ' + ' + str(self.mean_k_sigma) +  '*sigma')
		plt.plot(lower, 'r-', label='SMA' + str(self.MA_window) + ' -  ' + str(self.mean_k_sigma) +  '*sigma')
		plt.plot(buy_upper, 'y-', label='+' + str(self.buy_k_sigma) + '*sigma')
		plt.plot(buy_lower, 'y-', label='- ' + str(self.buy_k_sigma) + '*sigma')
		plt.title(buy_sell_signal + ' ' + symbol + ' || Beta : ' + str(beta) + ' || Alpha : ' + str(alpha))
		plt.legend(loc='best')

		if save_plot:
			plt.savefig('12/' + buy_sell_signal + '_'+str(symbol)+'.png')

		if show_plot:
			plt.show()
			plt.close()

	#Bollinger trading strategy : 
	def bollinger_buy_strategy(self, df, end_date, symbol, roll_mean, roll_std, upper, lower, beta, alpha):

		buy_df = pd.DataFrame(columns = ['BUY'], index = df.index.tolist())

		price = df.get_value(end_date, symbol)

		#buy_signal = (lower.get_value(end_date, symbol) - price)/lower.get_value(end_date, symbol)
		#print buy_signal


		buy_upper, buy_lower = self.bollinger_bands(lower, roll_std, self.buy_k_sigma)
			
		lbuy = buy_lower.loc[end_date]
		ubuy = buy_upper.loc[end_date]
		buy_price  = lower.loc[end_date]

		slope = (df[symbol].iloc[-1] - df[symbol].iloc[-3])

		# mu + 2 * sigma
		#Weaker Buy
		# mu - Buy
		#Stronger Buy
		# mu - 2 * sigma
		#Strongest Buy

		if price < ubuy and slope > 0:

			if price >= buy_price:

				#Buy Signal
				print 'Buy Signal for ' + symbol
				print 'Difference from lower band value ' + str(buy_price-price)
				self.plot_bollinger_bands(df, symbol, roll_mean, upper, lower, buy_upper, buy_lower, beta, alpha, 'Buy', show_plot=False, save_plot=True)

			elif price < buy_price and price > lbuy:

				#Stronger Buy Signal
				print 'Strong Buy Signal for ' + symbol
				print 'Difference from lower band value ' + str(buy_price-price)
				self.plot_bollinger_bands(df, symbol, roll_mean, upper, lower, buy_upper, buy_lower, beta, alpha, 'Strong_Buy', show_plot=False, save_plot=True)

			else:

				#Strongest Buy Signal
				print 'Very Strong Buy Signal for ' + symbol
				print 'Difference from lower band value ' + str(buy_price-price)
				self.plot_bollinger_bands(df, symbol, roll_mean, upper, lower, buy_upper, buy_lower, beta, alpha, 'Very_Strong_Buy', show_plot=False, save_plot=True)


	def bollinger_sell_strategy(self, df, end_date, symbol, roll_mean, roll_std, upper, lower, beta, alpha):

		sell_df = pd.DataFrame(columns = ['SELL'], index = df.index.tolist())
		
		price = df.get_value(end_date, symbol)

		#buy_signal = (lower.get_value(end_date, symbol) - price)/lower.get_value(end_date, symbol)
		#print buy_signal


		sell_upper, sell_lower = self.bollinger_bands(upper, roll_std, self.sell_k_sigma)
			
		lsell = sell_lower.loc[end_date]
		usell = sell_upper.loc[end_date]
		sell  = upper.loc[end_date]

		slope = (df[symbol].iloc[-1] - df[symbol].iloc[-3])

		#Strongest sell
		# mu + 2 * sigma
		#Stronger sell
		# mu - sell
		#weaker sell
		# mu - 2 * sigma
		

		if price > lsell and slope < 0:

			if price <= sell:

				#sell Signal
				print 'Sell Signal for ' + symbol
				print 'Difference from upper band value ' + str(sell-price)
				self.plot_bollinger_bands(df, symbol, roll_mean, upper, lower, sell_upper, sell_lower, beta, alpha, 'Sell', show_plot=False, save_plot=True)

			elif price > sell and price < usell:

				#Stronger Sell Signal
				print 'Strong Sell Signal for ' + symbol
				print 'Difference from upper band value ' + str(sell-price)
				self.plot_bollinger_bands(df, symbol, roll_mean, upper, lower, sell_upper, sell_lower, beta, alpha, 'Strong_Sell', show_plot=False, save_plot=True)

			else:

				#Strongest Sell Signal
				print 'Very Strong Sell Signal for ' + symbol
				print 'Difference from upper band value ' + str(sell-price)
				self.plot_bollinger_bands(df, symbol, roll_mean, upper, lower, sell_upper, sell_lower, beta, alpha, 'Very_Strong_Sell', show_plot=False, save_plot=True)




	#Take as input Alpha_Beta DF and Prices DF
	def Trading_Strategy(self, df, Alpha_Beta_DF):
		symbols = df.columns

		#print df

		#print Alpha_Beta_DF

		for symbol in symbols:
			roll_mean = df[symbol].rolling(window=self.MA_window, center=False).mean()
			roll_std  = df[symbol].rolling(window=self.MA_window, center=False).std()

			upper, lower = self.bollinger_bands(roll_mean, roll_std, self.mean_k_sigma)

			beta  = Alpha_Beta_DF.loc[symbol, 'Beta']
			alpha = Alpha_Beta_DF.loc[symbol, 'Alpha']
			#print beta, alpha

			trading_date = df.tail(1).index[0]
			#	print trading_date
			#self.Backtest(df, symbol, roll_mean, roll_std, upper, lower)

			self.bollinger_buy_strategy(df, str(trading_date), symbol, roll_mean, roll_std, upper, lower, beta, alpha)

			#self.bollinger_sell_strategy(df, str(trading_date), symbol, roll_mean, roll_std, upper, lower, beta, alpha)
