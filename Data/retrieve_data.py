import yfinance as yf

def retrieve_data(ticker):
    data = yf.download(ticker, start="2015-01-01", interval="1d",) #, end="2024-12-31"
    return data

def main():
    data = retrieve_data("SPY")
    print(data.head())
    data.to_csv("spy_data.csv")

if __name__ == "__main__":
    main()