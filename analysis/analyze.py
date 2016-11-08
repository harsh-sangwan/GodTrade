import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


nifty500 = pd.read_csv('../data/scripts/NSE/ind_nifty500list.csv')

#Import stock data from data directory
symbols = nifty500[nifty500.columns[2]]

def plot_returns(returns, symbol, ylabel):
	plt.plot(returns)
	plt.title(symbol)
	plt.ylabel('')
	plt.grid(True)
	plt.show()
	

#Index 0,1,2 - Daily, Monthly, Annual
def get_returns(data, symbol):
	prices = data['Adj Close']
	dates  = data['Date']

	daily   = prices.pct_change(1)
	monthly = prices.pct_change(21)
	annual  = prices.pct_change(252)
	
	print dates	
	plt.plot(dates, daily, color='blue', label='Daily')
	plt.plot(monthly, color='orange', label='Monthly')
	plt.plot(annual, color='green', label='Annually')
	plt.legend(loc='best')
	plt.title(symbol)
	plt.ylabel('Returns')
	plt.grid(True)
	plt.show()
	
	#print annual
	#return daily	

def analyze(data, symbol):
	prices = data['Adj Close']
	get_returns(data, symbol)
	
	#plt.plot(data['Date'], returns)
	#plt.show()

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		filename = symbol + ".csv"
		filepath = '../data/NSE/Cap-Size/' + filename
		filedata = pd.read_csv(filepath)
		analyze(filedata, symbol)
			
	except Exception, arg:
		continue
