import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt


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


def Monte_Carlo_Simulations(numPortfolios, numAssets, meanDailyReturns, covariance, numPeriodsAnnually, riskFreeRate):

	results = np.zeros((3,numPortfolios))

	print meanDailyReturns
	print covariance

	#Calculate portfolios

	for i in range(numPortfolios):
	    #Draw numAssets random numbers and normalize them to be the portfolio weights

	    weights = np.random.rand(numAssets)
	    weights /= np.sum(weights)

	    #print weights

	    #Calculate expected return and volatility of portfolio

	    pret, pvar = calcPortfolioPerf(weights, meanDailyReturns, covariance)

	    #Convert results to annual basis, calculate Sharpe Ratio, and store them

	    results[0,i] = pret*numPeriodsAnnually
	    results[1,i] = pvar*np.sqrt(numPeriodsAnnually)
	    results[2,i] = (results[0,i] - riskFreeRate)/results[1,i]


	return results


def Optimization(df, symbols):
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

	results = Monte_Carlo_Simulations(2500, numAssets, meanDailyReturns, covMatrix, 252, 0.04)

	################################
	##### Find optimal weights #####
	################################

	optimal_weights_df = pd.DataFrame(index=list(meanDailyReturns.index), columns=['Max_Sharpe_Ratio_Wts', 'Min_Var_Wts'])

	#Find portfolio with maximum Sharpe ratio
	riskFreeRate = 0.04

	maxSharpe = findMaxSharpeRatioPortfolio(weights, meanDailyReturns, covMatrix, riskFreeRate)
	#print maxSharpe['x']
	maxSharpeRET, maxSharpeSD = calcPortfolioPerf(maxSharpe['x'], meanDailyReturns, covMatrix)

	print maxSharpeRET
	print maxSharpeSD

	#Find portfolio with minimum variance

	minVar  = findMinVariancePortfolio(weights, meanDailyReturns, covMatrix)
	#print minVar['x']
	minVarRET, minVarSD = calcPortfolioPerf(minVar['x'], meanDailyReturns, covMatrix)

	print minVarRET
	print minVarSD

	optimal_weights_df['Max_Sharpe_Ratio_Wts'] = maxSharpe['x']
	optimal_weights_df['Min_Var_Wts'] = minVar['x']

	print optimal_weights_df

	#Find efficient frontier, annual target returns of 9% and 16% are converted to

	#match period of mean returns calculated previously

	#targetReturns = np.linspace(0.09, 0.16, 50)/(252./3)
	##efficientPortfolios = findEfficientFrontier(weights, meanDailyReturns, covMatrix, targetReturns)


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

	
	'''df = pd.DataFrame()	
	for i in xrange(len(symbols)):
		pret += weights[i] * returns[symbols[i]]
		df = pd.concat([df, returns[symbols[i]]], axis=1)

	df = pd.concat([df, pret], axis=1)
	print df'''