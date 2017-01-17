from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import os

url = 'http://www.moneycontrol.com/stocks/marketstats/indexcomp.php?optex=BSE&opttopic=indexcomp&index=12'

r = requests.get(url)
soup = bs(r.content, 'lxml')

for link in soup.findAll('a', {'class':'bl_12'}):
	print link