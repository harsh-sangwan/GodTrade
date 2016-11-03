import pandas_datareader.data as web
from datetime import datetime
import pandas as pd


start = datetime(1900, 1, 1)	#(YYYY, MM, DD)
end   = datetime(2016, 11, 3)

'''
#Download Nifty500 stocks
nse = pd.read_csv('NSE/ind_nifty500list.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
    try:
        symbol = symbols[rows]
        stock_symbol = symbol + ".NS"
        stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
        filename = '../NSE/Cap-Size/' + symbol + '.csv'
        stock_prices.to_csv(filename)
    except Exception, arg:
        print("Error downloading symbol:"+str(symbol))
        print(str(arg))
        continue
        
'''

#Download Nifty Indices
nse = pd.read_csv('NSE/nifty_indices.csv')
symbols = nse[nse.columns[1]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Indices/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - Indices ; symbol:"+str(symbol))
        	print(str(arg))


#Download Nifty Energy stocks
nse = pd.read_csv('NSE/ind_niftyenergylist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/Energy/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - Energy ; symbol:"+str(symbol))
        	print(str(arg))
'''

#Download Nifty Auto stocks
nse = pd.read_csv('NSE/ind_niftyautolist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/Auto/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading -Auto ; symbol:"+str(symbol))
        	print(str(arg))


#Download Nifty Bank stocks
nse = pd.read_csv('NSE/ind_niftybanklist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/Bank/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - Bank ; symbol:"+str(symbol))
        	print(str(arg))



#Download Nifty Finance stocks
nse = pd.read_csv('NSE/ind_niftyfinancelist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/Finance/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - Finance ; symbol:"+str(symbol))
        	print(str(arg))


#Download Nifty FMCG stocks
nse = pd.read_csv('NSE/ind_niftyfmcglist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/FMCG/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - FMCG ; symbol:"+str(symbol))
        	print(str(arg))


#Download Nifty IT stocks
nse = pd.read_csv('NSE/ind_niftyitlist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/IT/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - IT ; symbol:"+str(symbol))
        	print(str(arg))


#Download Nifty Media stocks
nse = pd.read_csv('NSE/ind_niftymedialist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/Media/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - Media ; symbol:"+str(symbol))
        	print(str(arg))


#Download Nifty Metal stocks
nse = pd.read_csv('NSE/ind_niftymetallist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/Metal/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - Metal ; symbol:"+str(symbol))
        	print(str(arg))

	
#Download Nifty Pharma stocks
nse = pd.read_csv('NSE/ind_niftypharmalist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/Pharma/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - Pharma ; symbol:"+str(symbol))
        	print(str(arg))


#Download Nifty Private-Bank stocks
nse = pd.read_csv('NSE/ind_nifty_privatebanklist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/Private_Bank/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - Private-Bank ; symbol:"+str(symbol))
        	print(str(arg))


#Download Nifty PSU-Bank stocks
nse = pd.read_csv('NSE/ind_niftypsubanklist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/PSU_Bank/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - PSU-Bank ; symbol:"+str(symbol))
        	print(str(arg))


#Download Nifty Realty stocks
nse = pd.read_csv('NSE/ind_niftyrealtylist.csv')
symbols = nse[nse.columns[2]]

for rows in range(0, len(symbols)):
	try:
		symbol = symbols[rows]
		stock_symbol = symbol + ".NS"
		stock_prices = web.DataReader(stock_symbol, "yahoo", start, end)
		filename = '../NSE/Sectors/Realty/' + symbol + '.csv'
		stock_prices.to_csv(filename)
	except Exception, arg:
        	print("Error downloading - Realty ; symbol:"+str(symbol))
        	print(str(arg))
'''

