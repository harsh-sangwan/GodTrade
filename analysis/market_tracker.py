import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_log_returns(stock):
	log_returns = []
	stock_price = list(stock['Close'])
	for i in range(0, (len(stock_price)-1)):
		log_returns.append(np.log(stock_price[i+1])-np.log(stock_price[i]))

	return log_returns


#Tells the highest return stock
def returns_vs_volatility(stocks, symbols, end_date, n):

	df = pd.DataFrame(columns = ['Start_Price', 'Mean_Price', 'End_Price', 'nday_Return', 'Mean_log_Return', 'Min_Day_Change', 'Max_Day_Change',  'Volatility'], 
					  index   = symbols)

	for stock_file in stocks:
		symbol = stock_file.split('/')[4][:-4]

		stock = pd.read_csv(stock_file)

		stock = stock.set_index('Date')
		stock = stock.loc[:end_date]
		stock.fillna(method='ffill', inplace=True)
		stock.fillna(method='bfill', inplace=True)
		stock = stock.reset_index()  
		stock = stock.tail(n)

		log_returns = get_log_returns(stock)

		#log_returns = sorted(log_returns)

		s = pd.Series({	 
				'Mean_Price'  	 : round(np.mean(list(stock['Close'])),2),
				'Volatility'	 : round(np.std(log_returns), 2), 
				'Start_Price' 	 : round(stock.iloc[0]['Close'],2), 
				'End_Price' 	 : round(stock.iloc[-1]['Close'],2),
				'nday_Return' 	 : round((np.log(stock.iloc[-1]['Close']) - np.log(stock.iloc[0]['Close'])),2),
				'Mean_log_Return': round(np.mean(log_returns),2),
				'Min_Day_Change' : round(min(log_returns),2),
				'Max_Day_Change' : round(max(log_returns),2)
			})

		df.loc[symbol] = s

	#print df
	df.index.name = 'Symbol'
	df = df.sort_values('nday_Return', ascending=False, axis=0)

	'''print end_date
	print df.head(10)
	print '\n'''
	output_file_name = 'Results_' + str(stock.iloc[0]['Date']) + '_' + str(stock.iloc[-1]['Date']) + '.csv'
	df.to_csv(output_file_name)

	return df
	
	
	'''ndf = df.head(20)
	t12 = list(ndf['Pct_Change'])
	t11 = ndf.index.tolist()
	plt.bar(range(len(t12)), t12, align='center')
	plt.xticks(range(len(t12)), t11, size='small')
	plt.ylabel('Percent Change')
	plt.title('Top growing stocks')
	plt.show()
	plt.close()'''
	
	#plt.plot(ndf.index.tolist(), ndf['Pct_Change'], 'b-')
	#Stocks = sorted(Stocks, key=lambda x : x['pct_change'])
	#print Stocks

def top_performers(df, top_n):
	top_df = df.head(top_n)
	print top_df.iloc[:,2:]
	#top_df['nday_Return'].hist(bins=20, label='Top Performers')
	#plt.show()


def worst_performers(df, worst_n):
	bottom_df = df.tail(worst_n)
	print bottom_df.iloc[:,2:]
