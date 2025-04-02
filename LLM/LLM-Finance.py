import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Define stock symbols
stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]

# Define time range for historical data
start_date = "2024-01-01"
end_date = "2024-03-20"

# Fetch stock price data
def fetch_stock_data(symbols, start, end):
    stock_data = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start, end=end)
        stock_data[symbol] = hist[['Open', 'Close', 'High', 'Low', 'Volume']]
        time.sleep(1)
    return stock_data

# Scrape Yahoo Finance News
def scrape_yahoo_stock_news(stock_symbol):
    url = f"https://finance.yahoo.com/quote/{stock_symbol}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    news_list = []
    for item in soup.find_all("h3"):
        headline = item.get_text()
        link = item.find("a")
        if link:
            link = "https://finance.yahoo.com" + link["href"]
        else:
            link = ""
        
        news_list.append({"headline": headline, "url": link})
    
    # Print sample news
    print("🔹 Sample News Headlines:", news_list[:5])
    
    return news_list

# Scrape news for all stocks
all_news = []
for stock in stocks:
    stock_news = scrape_yahoo_stock_news(stock)
    for news in stock_news:
        news["stock"] = stock  # Add stock symbol to data
    all_news.extend(stock_news)

# Convert news data to DataFrame and Save
news_df = pd.DataFrame(all_news)
if not news_df.empty:
    news_df.to_csv("yahoo_finance_news.csv", index=False)
    print("✅ All stock-specific financial news saved successfully.")
else:
    print("❌ No news articles found. Yahoo Finance structure may have changed.")

# Fetch stock price data
stock_data = fetch_stock_data(stocks, start_date, end_date)

# Convert stock data to DataFrame
stock_dfs = {symbol: df.reset_index() for symbol, df in stock_data.items()}

# Save stock data to CSV
for symbol, df in stock_dfs.items():
    df.to_csv(f"{symbol}_stock_data.csv", index=False)

print("Data collection complete. CSV files saved.")
