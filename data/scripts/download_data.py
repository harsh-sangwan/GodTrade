import pandas_datareader.data as web
from datetime import datetime
import pandas as pd
import os
from nsepy import get_history

def download_data(start_date, end_date, symbol, output_directory):
	try:
	        	
	    stock_symbol = symbol
	    #stock_prices = get_history(symbol=symbol, start=start_date, end=end_date)
	    stock_prices = web.DataReader(stock_symbol.encode('utf-8'), "yahoo", start_date, end_date)
	    output_file  = output_directory + symbol + '.csv'

	    if not os.path.exists(os.path.dirname(output_file)):
			try:
				os.makedirs(os.path.dirname(output_file))
				#stock_prices.to_csv(output_file)
			except OSError as exc:
				if exc.errno != errno.EEXIST:
					raise
	    		
	    stock_prices.to_csv(output_file)

	except Exception, arg:
	    print("Error downloading from yahoo ; symbol:"+str(symbol))
	    print(str(arg))



def save_data(start_date, end_date, symbol_file, symbols_colnum, mrkt, output_directory):
	df = pd.read_csv(symbol_file, header=0)
	print df
	symbols = df[df.columns[symbols_colnum]]
	
	for rows in range(0, len(symbols)):
		symbol = symbols[rows]
		download_data(start_date, end_date, str(symbol)+mrkt, output_directory)
		if rows%20 == 0:
			print 'Downloaded ' + str(symbol)+mrkt
		


if __name__ == "__main__":
 
 	now = datetime.now()
 	
	start = datetime(1900, 1, 1)	#(YYYY, MM, DD)
	end   = datetime(now.year, now.month, now.day)
	
	nse_cap_size_symbols = 'NSE/ind_nifty500list.csv'
	bse_cap_size_symbols = 'BSE/bse500.csv'

	nse_cap_size_op_dir	 = '../NSE/Cap-Size/'
	bse_cap_size_op_dir	 = '../BSE/Cap-Size/'

	nifty_index  = 'NSE/nifty_indices.csv'
	nifty_op_dir = '../NSE/Indices/'
	bse_op_dir = '../BSE/Indices/'

	nse_sectors = ['energy', 'auto', 'bank', 'finance', 'fmcg', 'it', 'media', 'metal', 'pharma', '_privatebank', 'psubank', 'realty'] 
	
	#Download Cap-Size data - NSE
	save_data(start, end, nse_cap_size_symbols, 2, '.NS', nse_cap_size_op_dir)
	
	#Download Nifty50
	download_data(start, end, '^NSEI', nifty_op_dir)

	#Download Nifty500
	download_data(start, end, '^CRSLDX', nifty_op_dir)

	#Download Sensex30
	download_data(start, end, '^BSESN', bse_op_dir)

	#Download Cap-Size data - BSE
	save_data(start, end, bse_cap_size_symbols, 3, '.BO', bse_cap_size_op_dir)

	#Download NSE sectors 
	#for i in xrange(len(nse_sectors)):
	#	save_data(start, end, 'NSE/ind_nifty'+str(nse_sectors[i])+'list.csv', 2, '../NSE/Sectors/'+nse_sectors[i]+'/')

	#Download Moneycontrol_Industries
	'''moneycontrol_path = "../NSE/Moneycontrol_Industries/"
	for name in os.listdir(moneycontrol_path):
		if name != ".DS_Store":
			save_data(start, end, moneycontrol_path+str(name)+'/stocks_list.csv', 1, moneycontrol_path+str(name)+'/')'''
