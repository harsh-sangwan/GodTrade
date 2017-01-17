from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import os
import numpy as np

url = 'http://www.moneycontrol.com/stocks/marketstats/indexcomp.php?optex=BSE&opttopic=indexcomp&index=12'

r = requests.get(url)
soup = bs(r.content, 'lxml')

numberOfRows = 504
bse = pd.DataFrame(index=np.arange(0, numberOfRows), columns=('Company', 'Symbol'))

ind = 0
for link in soup.findAll('a', {'class':'bl_12'}):
	if '/india/' in link['href']:
		symbol_url = 'http://www.moneycontrol.com' + link['href']
		company = link.text

		r2 = requests.get(symbol_url)
		soup2 = bs(r2.content, 'lxml')

		for t in soup2.findAll('div', {'class':'FL gry10'}):
			t = list(str(t).split(' '))
			symbol = t[3]


			s = pd.Series({'Company'  : company,  
				   		   'Symbol'   : symbol})

			bse.loc[ind] = s
			ind = ind+1

print bse
bse.to_csv('bse500.csv')

	


			

