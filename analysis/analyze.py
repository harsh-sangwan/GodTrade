import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import numpy as np
import talib
from datetime import datetime, timedelta
import os
import scipy.optimize as sco

def plotting_dates_and_prices(dates, close, symbol):
	dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
	fig = plt.figure(figsize=(18, 18))
	plt.plot_date(dates, close, 'b-')
	plt.grid()
	plt.xlabel('Dates')
	plt.ylabel('Closing Prices')
	plt.title(symbol)
	
	return plt

def plot_time_frame(df, start, end, symbol):
	close = df.loc[start:end]['Close']
	dates = df.loc[start:end].index.tolist()
	plotting_dates_and_prices(dates, close, symbol)
	
def plot_last_ndays(df, n, symbol):
	df = df.tail(n)
	close = df['Close']
	dates = df.index.tolist()
	plotting_dates_and_prices(dates, close, symbol)
	

#Tells the highest return stock
def highest_return(stocks, symbols, end_date, n, output_file_name):

	df = pd.DataFrame(columns = ['Start_Date', 'End_Date', 'Mean_Price',  'Volatility', 'Start_Price', 'End_Price', 'nday_Return', 'Mean_log_Return', 'Min_Day_Change', 'Max_Day_Change'], 
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

		log_returns = []
		stock_price = list(stock['Close'])
		for i in range(0, (len(stock_price)-1)):
			log_returns.append(np.log(stock_price[i+1])-np.log(stock_price[i]))

		log_returns = sorted(log_returns)

		s = pd.Series({	
				'Start_Date' 	 : stock.iloc[0]['Date'], 
				'End_Date'   	 : stock.iloc[-1]['Date'], 
				'Mean_Price'  	 : round(np.mean(stock_price),2),
				'Volatility'	 : round(np.std(log_returns), 2), 
				'Start_Price' 	 : round(stock.iloc[0]['Close'],2), 
				'End_Price' 	 : round(stock.iloc[-1]['Close'],2),
				'nday_Return' 	 : round((np.log(stock.iloc[-1]['Close']) - np.log(stock.iloc[0]['Close'])),2),
				'Mean_log_Return': round(np.mean(log_returns),2),
				'Min_Day_Change' : round(log_returns[0],2),
				'Max_Day_Change' : round(log_returns[-1],2)
			})

		df.loc[symbol] = s

	#print df
	df.index.name = 'Symbol'
	df = df.sort_values('nday_Return', ascending=False, axis=0)

	'''print end_date
	print df.head(10)
	print '\n'''
	df.to_csv(output_file_name)
	
	
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
		
def compare_rankings(old_file, new_file, symbols):
	old_df = pd.read_csv(old_file)
	new_df = pd.read_csv(new_file)
	out_df = pd.DataFrame(index = symbols, columns = ['Old_Rank', 'New_Rank', 'Difference'])

	old_df = old_df.set_index('Symbol')
	new_df = new_df.set_index('Symbol')

	for symbol in symbols:
		old_index  = old_df.index.get_loc(symbol)
		new_index  = new_df.index.get_loc(symbol)
		difference = old_index - new_index

		series = pd.Series({'Old_Rank':old_index, 'New_Rank':new_index, 'Difference':difference})
		out_df.loc[symbol] = series

	out_df = pd.concat([out_df, old_df, new_df], axis=1)
	out_df = out_df.sort_values('Difference', ascending=False, axis=0)
	out_df.to_csv('Rankings.csv')
		

#Save a df of all stocks close for a time range
def compare_stocks(file_list, start, end, nifty50, nifty500):
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

def bollinger_bands(rmean, rstd, factor):
	upper = rmean + factor * rstd
	lower = rmean - factor * rstd

	return upper, lower

def compare_hists(df, symbol1, symbol2):
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
	plt.show()
	plt.close()

	print rets.kurtosis()

def plot_bollinger_bands(df, symbol, roll_mean, upper, lower, buy_upper, buy_lower, beta):
	dates = df.index.tolist()
	plt = plotting_dates_and_prices(dates, df[symbol], symbol)
	plt.plot(roll_mean, 'g-', label='SMA20')
	plt.plot(upper, 'r-', label='+2*sigma')
	plt.plot(lower, 'r-', label='-2*sigma')
	plt.plot(buy_upper, 'y-', label='0.3')
	plt.plot(buy_lower, 'y-', label='0.3')
	plt.title(symbol + ' Beta : ' + str(beta))
	plt.legend(loc='best')
	plt.savefig('12/'+str(symbol)+'.png')
	#plt.show()
	#plt.close()

#Bollinger trading strategy : 
def bollinger_trading_strategy(df, end_date, symbol, roll_mean, roll_std, upper, lower, factor, beta):
	price = df.get_value(end_date, symbol)

	#buy_signal = (lower.get_value(end_date, symbol) - price)/lower.get_value(end_date, symbol)
	#print buy_signal


	buy_upper, buy_lower = bollinger_bands(lower, roll_std, factor)
		
	lbuy = buy_lower.loc[end_date]
	ubuy = buy_upper.loc[end_date]

	#slope = (df[symbol].iloc[-1] - df[symbol].iloc[-3])


	if price < ubuy:# and slope > 0:
		print symbol + '\t' + str(lower.loc[end_date]-price)
		plot_bollinger_bands(df, symbol, roll_mean, upper, lower, buy_upper, buy_lower, beta)


#Find alpha and beta between two stocks
def alpha_beta(df, s1, s2):
	rets1 = df[s1].pct_change(1)
	rets2 = df[s2].pct_change(1)

	rets1 = rets1.fillna(0)
	rets2 = rets2.fillna(0)

	ndf = pd.DataFrame({s1:rets1, s2:rets2})


	ndf.plot(kind='scatter', x=s1, y=s2)
	beta_y, alpha_y = np.polyfit(rets1, rets2, 1)

	print 'beta  : ' + str(beta_y)
	print 'alpha : ' + str(alpha_y)
	print ndf.corr(method='pearson')

	plt.plot(rets1, beta_y*rets1 + alpha_y, '-', color='r')
	plt.title('Compare Returns')
	plt.show()



def calcPortfolioPerf(weights, meanDailyReturns, covariance):
	portReturn = np.sum(meanDailyReturns * weights)
	portStdDev = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

	return portReturn, portStdDev

def negSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate):
    '''
    Returns the negated Sharpe Ratio for the speicified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    p_ret, p_var = calcPortfolioPerf(weights, meanReturns, covMatrix)

    return -(p_ret - riskFreeRate) / p_var

def getPortfolioVol(weights, meanReturns, covMatrix):
    '''
    Returns the volatility of the specified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    The portfolio's volatility
    '''
    return calcPortfolioPerf(weights, meanReturns, covMatrix)[1]

def getPortfolioReturn(weights, meanReturns, covMatrix):

    return calcPortfolioPerf(weights, meanReturns, covMatrix)[0]

def findMinVariancePortfolio(weights, meanReturns, covMatrix):
    '''
    Finds the portfolio of assets providing the lowest volatility

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0,1) for asset in range(numAssets))

    
    opts = sco.minimize(getPortfolioVol, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return opts

def findEfficientReturn(weights, meanReturns, covMatrix, targetReturn):
    '''
    Finds the portfolio of assets providing the target return with lowest
    volatility

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    targetReturn: APR of target expected return

    OUTPUT
    Dictionary of results from optimization
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)

    constraints = ({'type': 'eq', 'fun': lambda x: getPortfolioReturn(x) - targetReturn},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    bounds = tuple((0,1) for asset in range(numAssets))

    return sco.minimize(getPortfolioVol, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

def findEfficientFrontier(weights, meanReturns, covMatrix, rangeOfReturns):
    '''
    Finds the set of portfolios comprising the efficient frontier

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    targetReturn: APR of target expected return

    OUTPUT
    Dictionary of results from optimization
    '''
    efficientPortfolios = []
    for ret in rangeOfReturns:
        efficientPortfolios.append(findEfficientReturn(weights, meanReturns, covMatrix, ret))

    return efficientPortfolios

def findMaxSharpeRatioPortfolio(weights, meanReturns, covMatrix, riskFreeRate):
    '''
    Finds the portfolio of assets providing the maximum Sharpe Ratio

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0,1) for asset in range(numAssets))

    opts = sco.minimize(negSharpeRatio, numAssets*[1./numAssets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return opts

def Monte_Carlo_simulations(numPortfolios, numAssets, meanDailyReturn, covariance, numPeriodsAnnually, riskFreeRate):

	results = np.zeros((3,numPortfolios))

	print meanDailyReturn
	print covariance

	#Calculate portfolios

	for i in range(numPortfolios):
	    #Draw numAssets random numbers and normalize them to be the portfolio weights

	    weights = np.random.rand(numAssets)
	    weights /= np.sum(weights)

	    #print weights

	    #Calculate expected return and volatility of portfolio

	    pret, pvar = calcPortfolioPerf(weights, meanDailyReturn, covariance)

	    #Convert results to annual basis, calculate Sharpe Ratio, and store them

	    results[0,i] = pret*numPeriodsAnnually
	    results[1,i] = pvar*np.sqrt(numPeriodsAnnually)
	    results[2,i] = (results[0,i] - riskFreeRate)/results[1,i]


	################################
	##### Find optimal weights #####
	################################

	optimal_weights_df = pd.DataFrame(index=list(meanDailyReturn.index), columns=['Max_Sharpe_Ratio_Wts', 'Min_Var_Wts'])

	#Find portfolio with maximum Sharpe ratio

	maxSharpe = findMaxSharpeRatioPortfolio(weights, meanDailyReturn, covariance, riskFreeRate)
	#print maxSharpe['x']
	maxSharpeRET, maxSharpeSD = calcPortfolioPerf(maxSharpe['x'], meanDailyReturn, covariance)

	print maxSharpeRET
	print maxSharpeSD

	#Find portfolio with minimum variance

	minVar  = findMinVariancePortfolio(weights, meanDailyReturn, covariance)
	#print minVar['x']
	minVarRET, minVarSD = calcPortfolioPerf(minVar['x'], meanDailyReturn, covariance)

	print minVarRET
	print minVarSD

	optimal_weights_df['Max_Sharpe_Ratio_Wts'] = maxSharpe['x']
	optimal_weights_df['Min_Var_Wts'] = minVar['x']

	print optimal_weights_df

	#Find efficient frontier, annual target returns of 9% and 16% are converted to

	#match period of mean returns calculated previously

	#targetReturns = np.linspace(0.09, 0.16, 50)/(252./3)
	##efficientPortfolios = findEfficientFrontier(weights, meanDailyReturn, covariance, targetReturns)

	#print efficientPortfolios
	#print results[4:]


	plt.plot(results[1,], results[0,], marker='o')
	plt.plot(maxSharpeSD, maxSharpeRET, marker='*', color='g', markersize='12')
	plt.plot(minVarSD, minVarRET, marker='*', color='r', markersize='12')

	plt.xlabel('Expected Volatility')
	plt.ylabel('Expected Return')

	plt.ylim([min(results[0,])-1,max(results[0,])+1])
	plt.xlim([min(results[1,])-1,max(results[1,])+1])
	#plt.colorbar(label='Sharpe Ratio')

	#portfolio_label = ", ".join(list(meanDailyReturn.index))
	#	plt.title('Portfolio of ' + portfolio_label)
	plt.savefig('12/Monte_carlo.png')


def portfolio_optimization(df, symbols):
	returns = df.pct_change(1)
	returns = returns.fillna(0)

	#Initialize weights as equal  for all
	weights = np.array([1/float(len(symbols)) for x in symbols])
	returns = returns[symbols]

	#See the Dataframe
	print returns
	
	meanDailyReturns = returns.mean()
	covMatrix = returns.cov()

	#pret, pstd = calcPortfolioPerf(weights, meanDailyReturns, covMatrix)

	numAssets = len(symbols)

	Monte_Carlo_simulations(2500, numAssets, meanDailyReturns, covMatrix, 252, 0.04)



	
	'''df = pd.DataFrame()	
	for i in xrange(len(symbols)):
		pret += weights[i] * returns[symbols[i]]
		df = pd.concat([df, returns[symbols[i]]], axis=1)

	df = pd.concat([df, pret], axis=1)
	print df'''

# [ Input <- output of compare_stocks ]
def compare_statistics(df, start_date, end_date):
	df = df.set_index('Date')
	df = df.loc[start_date:end_date]

	symbols = list(df.columns)

	#Global Statistics
	mean   = df.mean()
	median = df.median()
	std    = df.std()

	global_df = pd.concat([mean, median, std], axis=1)
	global_df = global_df.rename(columns={'0':'Mean', '1':'Median', '2':'Std'})
	global_df.to_csv('12/Global_Statistics.csv')

	rolling_df = pd.DataFrame(columns=symbols)

	

	market_rets = df['Nifty50'].pct_change(1)
	market_rets = market_rets.fillna(0)
	#print market_rets


	ABDF = pd.DataFrame(index=symbols, columns=['Beta', 'Alpha'])

	#Rolling statistics
	Moving_Average_Window = 10
	
	for symbol in symbols:
		roll_mean = df[symbol].rolling(window=Moving_Average_Window, center=False).mean()
		roll_std  = df[symbol].rolling(window=Moving_Average_Window, center=False).std()


		upper, lower = bollinger_bands(roll_mean, roll_std, 2)

		#Bollinger trading strategy  
		#bollinger_trading_strategy(df, end_date, symbol, roll_mean, roll_std, upper, lower, 0.5)
		
		rolling_df[symbol] = roll_mean

		if symbol != 'Nifty50' and symbol != 'Nifty500':
			stock_rets = df[symbol].pct_change(1)
			stock_rets = stock_rets.fillna(0)
			#print stock_rets
			beta, alpha = np.polyfit(market_rets, stock_rets, 1)

			
			bollinger_trading_strategy(df, end_date, symbol, roll_mean, roll_std, upper, lower, 0.5, beta)


			ABDF.loc[symbol] = pd.Series({'Beta':round(beta, 2), 'Alpha':round(alpha, 2)})


	ABDF.to_csv('12/alpha_beta.csv')

	#print df['MINDTREE.NS']
	
	#Portfolio Optimization
	test_symbols = ['MINDTREE.NS', 'PETRONET.NS', 'SUPREMEIND.NS', 'MERCK.NS', 'TRENT.NS']
	portfolio_optimization(df, test_symbols)

	#Comapre two stocks scatter plots
	'''symbol1 = 'Nifty50'
	symbol2 = 'INFY'
	alpha_beta(df, symbol1, symbol2)'''





if __name__ == "__main__":

	now = datetime.now()
	start_date = '2016-12-10'
	#end_date   = str(now.year)+'-'+str(now.month)+'-'+str(now.day)
	end_date   = '2017-01-11'


	cap_path = '../data/NSE/Cap-Size/'
	cap_stocks = []
	symbols = []

	#Nifty50 data
	nifty50 = pd.read_csv('../data/NSE/Indices/^NSEI.csv')

	#Nifty500 data
	nifty500 = pd.read_csv('../data/NSE/Indices/^CRSLDX.csv')
	#print len(os.listdir(cap_path))

	for filename in os.listdir(cap_path):
		symbol = filename[:-4]
		symbols.append(str(symbol))

		if filename != '.DS_Store':
			df = pd.read_csv(cap_path+str(filename))
			#print df
			df = df.set_index('Date')

			close = df.loc[start_date:end_date]['Close']


			cap_stocks.append(cap_path+str(filename))

		#plot_time_frame(df, start_date, end_date, symbol)
		#plot_last_ndays(df, 100, symbol)

	#highest_return(cap_stocks, symbols, end_date, 35, '10/Results_'+str(end_date)+'.csv')
	compare_stocks(cap_stocks, start_date, end_date, nifty50, nifty500)

	#compare_statistics


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





