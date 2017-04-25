import pandas as pd
import talib
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../data/NSE/Cap-Size/SUPREMEIND.NS.csv')

def create_feature_df(df):

	ema30 = pd.Series(talib.EMA(np.array(df['Close']), timeperiod=30), name='EMA30')
	#print ema30

	sma20  = pd.Series(talib.SMA(np.array(df['Close']), timeperiod=20),  name='SMA20')
	sma50  = pd.Series(talib.SMA(np.array(df['Close']), timeperiod=50),  name='SMA50')
	sma200 = pd.Series(talib.SMA(np.array(df['Close']), timeperiod=200), name='SMA200')


	#trading strategies
	upperband, middleband, lowerband = talib.BBANDS(np.array(df['Close']), timeperiod=5, nbdevup=2, nbdevdn=2)

	upperband  = pd.Series(upperband, name='Bollinger5+2')
	middleband = pd.Series(middleband, name='Bollinger5')
	lowerband  = pd.Series(lowerband, name='Bollinger5-2')

	BBand = pd.Series((df['Close'] - middleband)/(2*(upperband-middleband)), name='BBAND')

	macd, macdsignal, macdhist = talib.MACD(np.array(df['Close']), fastperiod=12, slowperiod=26, signalperiod=9)

	macd       = pd.Series(macd, name='MACD')
	macdsignal = pd.Series(macdsignal, name='MACD_Signal')
	macdhist   = pd.Series(macdhist, name='MACD_Hist')

	obv = pd.Series(talib.OBV(np.array(df['Close']), np.array(df['Volume'].astype(float))), name='OBV').astype(int)
	mfi = pd.Series(talib.MFI(np.array(df['High']), np.array(df['Low']), np.array(df['Close']), np.array(df['Volume'].astype(float)), timeperiod=14), name='MFI')
	rsi = pd.Series(talib.RSI(np.array(df['Close']), timeperiod=14), name='RSI')
	adx = pd.Series(talib.ADX(np.array(df['High']), np.array(df['Low']), np.array(df['Close']), timeperiod=14), name='ADX')
	wpr = pd.Series(talib.WILLR(np.array(df['High']), np.array(df['Low']), np.array(df['Close']), timeperiod=14), name='WILLR')

	daily = pd.Series(df['Close'].pct_change(1), name='DAILY_RETURN')
	oc_pct = pd.Series(((df['Open']+df['Close'])/2).pct_change(1), name='OC_DAILY_RETURN')
	df = df.join([sma20, sma50, sma200, BBand, macd, macdsignal, macdhist, obv, mfi, rsi, adx, wpr, daily, oc_pct])

	return df

'''def run_regression(df):
def run_random_forest(df):
def run_decision_tree(df):
'''	


df = create_feature_df(df)
'''df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.dropna()'''
#print df['OBV']
#print(df.corr())
#result = seasonal_decompose(df['OBV'], model='additive', freq=252)
#result.plot()
#df['Close'].plot()
#plt.show()
#df.to_csv('ml.csv')

prices = list(df['Close'])
dates = list(df['Date'])

long_orders = []
i = 0
while i < len(prices)-1:
	while i < len(prices)-1 and prices[i+1] <= prices[i]:
		i = i + 1
	if i == len(prices)-1:
		break
	buy_on = dates[i]
	buy_at = prices[i]
	i = i + 1
	while i < len(prices)-1 and prices[i-1] <= prices[i]:
		i = i + 1
	order = ({	'Buy_On'  : buy_on,
				'Buy_At'  : buy_at,
				'Sell_On' : dates[i-1],
			  	'Sell_At' : prices[i-1],
			  	'Profit'  : prices[i-1] - buy_at})
	long_orders.append(order)

short_orders = []
i = 1
while i < len(prices)-1:
	while i < len(prices)-1 and prices[i] <= prices[i+1]:
		i = i + 1
	if i == len(prices)-1:
		break
	short_on = dates[i]
	short_at = prices[i]
	i = i + 1
	while i < len(prices)-1 and prices[i-1] >= prices[i]:
		i = i + 1
	order = ({	'Buy_On'   : dates[i-1],
				'Buy_At'   : prices[i-1],
				'Short_On' : short_on,
			  	'Short_At' : short_at,
			  	'Profit'   : short_at - prices[i-1]})
	short_orders.append(order)

worth = 10000
min_profit = 500
long_orders = [order for order in long_orders if (math.ceil(worth/order['Buy_At'] * order['Profit']) >= min_profit)]
#for i in range(0, len(long_orders)):
#	print(long_orders[i])

short_orders = [order for order in short_orders if (math.ceil(worth/order['Buy_At'] * order['Profit']) >= min_profit)]

Orders_DF = pd.DataFrame(index=dates, columns=['Long', 'Short'])

for i in range(0, len(long_orders)):
	Orders_DF.set_value(long_orders[i]['Buy_On'], 'Long', -1)
	Orders_DF.set_value(long_orders[i]['Sell_On'], 'Long', +1)

for i in range(0, len(short_orders)):
	Orders_DF.set_value(short_orders[i]['Buy_On'], 'Short', -1)
	Orders_DF.set_value(short_orders[i]['Short_On'], 'Short', +1)

Orders_DF = Orders_DF.fillna(0)

df = df.set_index('Date')
df = df.join(Orders_DF)

df = df.dropna()
df = df.astype(int)

X = df[['Close', 'Volume', 'SMA20', 'SMA50', 'SMA200', 'BBAND', 'MACD', 'MACD_Signal', 'OBV', 'MFI', 'RSI', 'ADX', 'WILLR']]
Y = df['Long']
model= RandomForestClassifier(n_estimators=1000)
# Train the model using the training sets and check score

fit = model.fit(X,Y)

plt.plot()
plt.show()

#for i in range(0, len(short_orders)):
#	print(short_orders[i])

#add market index
#add_index_Close()



