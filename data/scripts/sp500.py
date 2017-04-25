from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import os
import numpy as np

url = 'http://www.moneycontrol.com/stocks/marketstats/indexcomp.php?optex=BSE&opttopic=indexcomp&index=12'

r = requests.get(url)
soup = bs(r.content, 'lxml')

numberOfRows = 504
bse = pd.DataFrame(index=np.arange(0, numberOfRows), columns=('Company', 'BSE_Symbol', 'NSE_Symbol'))

ind = 0
for link in soup.findAll('a', {'class':'bl_12'}):
	if '/india/' in link['href']:
		symbol_url = 'http://www.moneycontrol.com' + link['href']
		company = link.text

		r2 = requests.get(symbol_url)
		soup2 = bs(r2.content, 'lxml')


		for t in soup2.findAll('div', {'class':'FL gry10'}):
			t = list(str(t).split(' '))
			bse_symbol = t[3]
			nse_symbol = t[t.index('NSE:')+1]
			nse_symbol = nse_symbol.replace('&amp;','&')

			if ind%20 == 0:
				print 'Downloaded .. ' + nse_symbol

			s = pd.Series({'Company'  	: company,  
				   		   'BSE_Symbol' : bse_symbol,
				   		   'NSE_Symbol' : nse_symbol})

			bse.loc[ind] = s
			ind = ind+1

#print bse
bse = bse.dropna()
print bse
bse.to_csv('BSE/bse500.csv')

	


			

