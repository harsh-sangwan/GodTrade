import pandas as pd

#Save a df of all stocks close for a time range
#Club the Nifty50 and Nifty500 index also
def compare_prices(file_list, start, end, nifty50, nifty500):
	temp_df = pd.read_csv(file_list[0])
	temp_df = temp_df.set_index('Date')

	symbol = file_list[0].split('/')[-1][:-4]

	df = pd.DataFrame()
	df = pd.concat([df, temp_df.loc[start:end]['Close']], axis=1)
	df = df.rename(columns={'Close':symbol})

	#dates = pd.date_range(start,end_date)
	#DF = pd.DataFrame(index=dates)

	#print DF
	for stock_file in file_list:
		symbol = stock_file.split('/')[-1][:-4]
		
		stock = pd.read_csv(stock_file)
		stock = stock.set_index('Date')
		close = stock.loc[start:end]['Close']

		df = pd.concat([df, close], axis=1)
		df = df.rename(columns={'Close':symbol})

		#DF = DF.join(close)

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

	df.fillna(method='ffill', inplace=True)
	df.fillna(method='bfill', inplace=True)
	print df
	df.to_csv('12/All_Stocks.csv')
	#print DF
	
	df = pd.read_csv('12/All_Stocks.csv')
	start = '2017-01-06'
	compare_statistics(df, start, end)