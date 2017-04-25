import numpy as np
import pandas as pd


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