# GodTrade
Find the best entry and exit strategies

1. Get updated financial data of stocks on NSE and BSE. Will be maintained in data/ directory

Under the directory .. 
-> Two folders - NSE & BSE - contains subfolders and Sensex25 & Nifty50 prices
-> then further subfolders sector-wise (Auto, Pharma, IT etc..)
-> where each subfolder will have all the stocks of that sector..
->* can have scripts directory in data/ directory which will have python/R scripts that would update the daily price data for all stocks.


2. How does the latest news affect the movement of stock prices? Will be maintained in the news/ directory

->* Under this directory will also have scripts/ directory that would fetch top news from ET.
-> SENTIMENT ANALYSIS : Next figure out if the news is a +ve/-ve sentiment for the stock .. eg : recent Tata Sons news .. 
 .. if the news is a general market mover eg : Brexit, GST news....
-> sentiment analysis scripts will also be added in scripts directory of news i.e. news/scripts


3. Identify Technical Analysis patterns .. will be added in TA/ directory..further .. patterns/ directory..

-> Currently some work has been done on the pattern script that identifies TA patterns throughout the data.. 
-> Signal enhancements need to be done i.e. how good a particular pattern is .. 
-> further monitor the trade daily and give the best entry and exit points .. 

Some good material on TA : 

-> http://zerodha.com/varsity/
-> http://zerodha.com/z-connect/traders-zone/review-books/ncfm-workbook-on-technical-analysis-a-must-read
-> Technical Analysis of the Financial Markets: A Comprehensive Guide to Trading Methods and Applications (New York Institute of Finance) [BIBLE]

As we keep on reading through material we'll keep adding more functionalities to the TA code
TA can further be used with commodities, currencies...

4. How much are the stocks correlated to the sector and SENSEX/NIFTY

-> On getting the data, correlation has to be found as to find if positive correlations (move together) or negative correlations (move oppositely)


**As we keep getting more insights & knowledge into the trading world, we will keep adding more functionalities

** Our FINAL aim is to actively trade(seconds) throughout multiple markets[Commodities, Currencies, F&O, Equities](globally) through all sectors.. and we need to accordingly build good algorithms .. 'coz we are too lazy and don't trust our psychology that much to trade on our own .. :P

*** TRADING is a RISKY BUSINESS .. IT's a business and should be treated accordingly..


Some good books to read .. 

-> Building Algorithmic Trading Systems: A Trader′s Journey From Data Mining to Monte Carlo Simulation to Live Trading + Website (Wiley Trading) - [for ALGO TRADING]
-> Algorithmic Trading: Winning Strategies and Their Rationale (Wiley Trading)
-> Market Wizards: Interviews With Top Traders Updated
-> How to Make Money in Stocks: A Winning System in Good Times and Bad
-> Reminiscences of a Stock Operator [for Trading in general]
-> Common Stocks and Uncommon Profits and Other Writings
-> https://www.quora.com/What-is-the-best-recommened-starter-guide-or-book-for-Algo-Trading

As we pick up pace on TA and make some profits using it. We'll head to commodities, currencies .. & then further F&O
 
 


