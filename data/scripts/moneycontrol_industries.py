from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import os


def download_stock_list(sector_name, sector_link):
			
	sector_stocks = '../NSE/Moneycontrol_Industries/'+str(sector_name)+'/stocks_list.csv'

	if not os.path.exists(os.path.dirname(sector_stocks)):
		try:
			os.makedirs(os.path.dirname(sector_stocks))

		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise

	sector_request = requests.get(sector_link)
	sector_soup    = bs(sector_request.content, 'lxml')

	stocks = []

	for spans in sector_soup.findAll('span', {'class':'gld13 disin'}):
		for company in spans.find_all('a'):
			shitty_link = company['href']
			stock_name = company.string

			if 'javascript' not in shitty_link:
				company_link = shitty_link

				company_request = requests.get(company_link)
				company_soup    = bs(company_request.content, 'lxml')

				for t in company_soup.findAll('div', {'class':'FL gry10'}):
					t = list(str(t).split(' '))
					stock_symbol = t[t.index('NSE:')+1]
					stock_symbol = stock_symbol.replace('&amp;','&')

					print sector_name + '\t' + stock_name + '\t\t' + stock_symbol
							
					stock = {'Stock_Name':stock_name, 'Stock_Symbol':stock_symbol}
					stocks.append(stock)

	stock_df = pd.DataFrame(stocks)
	stock_df.to_csv(sector_stocks)


url = 'http://www.moneycontrol.com/stocks/marketstats/industry-classification/nse/vanaspati-oils.html'

download_stock_list('Vanapati - Oils', url)

r = requests.get(url)
soup = bs(r.content, 'lxml')

for lis in soup.findAll('ul', {'class':'stat_lflist'}):
	for links in lis.find_all('a'):
		sector_link = links['href']
		sector_link = 'http://www.moneycontrol.com' + sector_link

		sector = links.string

		download_stock_list(sector, sector_link)


