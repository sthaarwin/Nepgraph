import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import date, datetime, timedelta

ROOT_URL = 'https://www.nepalstock.com'




class DataManager:
    def __init__(self, data_path='data/nepse_prices.csv'):
        if not os.path.isabs(data_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_path = os.path.join(base_dir, data_path)
        else:
            self.data_path = data_path
        self.raw_data = None
        self.log_returns = None

    def get_data(self, tickers=None, start_date=None, end_date=None, force_fetch=False):
        if not force_fetch and os.path.exists(self.data_path):
            print(f"Loading data from local file: {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path, index_col='date', parse_dates=True)
            self.raw_data.index = pd.to_datetime(self.raw_data.index).date
            print("Data loaded successfully from CSV.")
            return self.raw_data

        print("Local data not found or `force_fetch` is True. Fetching from local repo...")
        return self._fetch_from_local_repo(tickers, start_date, end_date)

    def _fetch_from_local_repo(self, tickers, start_date, end_date):
        repo_path = os.path.join(os.path.dirname(__file__), 'nepse-data', 'data', 'company-wise')
        if tickers is None:
            tickers = [f.split('.')[0] for f in os.listdir(repo_path) if f.endswith('.csv')]
            
        print(f"Fetching data from local repo for {len(tickers)} tickers...")
        
        all_data = []
        for i, ticker in enumerate(tickers):
            ticker = ticker.upper()
            csv_file = os.path.join(repo_path, f"{ticker}.csv")
            
            if not os.path.exists(csv_file):
                print(f"Ticker {ticker} CSV not found in {repo_path}")
                continue
                
            try:
                print(f"({i+1}/{len(tickers)}) Fetching {ticker}...")
                df = pd.read_csv(csv_file)
                if 'published_date' in df.columns and 'close' in df.columns:
                    df['date'] = pd.to_datetime(df['published_date']).dt.date
                    df = df[['date', 'close']]
                    df.rename(columns={'close': ticker}, inplace=True)
                    df.set_index('date', inplace=True)
                    df = df[~df.index.duplicated(keep='last')]
                    all_data.append(df)
            except Exception as e:
                print(f"Could not fetch data for {ticker}. Reason: {e}")

        if not all_data:
            print("No data was fetched from local repo. Creating dummy data instead...")
            return self._create_dummy(tickers)

        combined_df = pd.concat(all_data, axis=1)
        combined_df.sort_index(inplace=True)
        combined_df.bfill(inplace=True)
        combined_df.ffill(inplace=True)

        if start_date:
            combined_df = combined_df[combined_df.index >= pd.to_datetime(start_date).date()]
        if end_date:
            combined_df = combined_df[combined_df.index <= pd.to_datetime(end_date).date()]

        self.raw_data = combined_df
        print("Historical data fetched and cleaned successfully.")
        
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        self.raw_data.to_csv(self.data_path)
        print(f"Data saved to {self.data_path}")
            
        return self.raw_data

    def _create_dummy(self, tickers, days=365):
        print(f"Creating dummy data for {len(tickers)} tickers...")
        dates = pd.to_datetime(pd.date_range(end=date.today(), periods=days))
        data = {'date': dates}
        for ticker in tickers:
            price = 100 + np.random.randint(-10, 10)
            prices = [price]
            for _ in range(1, len(dates)):
                move = np.random.normal(0, 2)
                price += move
                prices.append(max(10, price))
            data[ticker.upper()] = prices
        
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)
        print(f"Dummy data saved to {self.data_path}")
        
        return self.get_data(tickers=tickers)

    def calculate_log_returns(self):
        if self.raw_data is None:
            print("Error: Raw data is not loaded. Please get data first.")
            return None
        
        self.log_returns = np.log(self.raw_data / self.raw_data.shift(1))
        self.log_returns = self.log_returns.dropna()
        print("Log returns calculated.")
        return self.log_returns


if __name__ == '__main__':
    TARGET_TICKERS = None
    DATA_FILE = 'data/nepse_prices.csv'
    
    # We must ensure creating directory if running from non-root
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    
    manager = DataManager(data_path=DATA_FILE)
    data = manager.get_data(tickers=TARGET_TICKERS, force_fetch=False)

    if data is not None:
        log_returns = manager.calculate_log_returns()
        if log_returns is not None:
            print("\n--- Calculated Log Returns (Sample) ---")
            print(log_returns.head())
            print("\n--- Data Description ---")
            print(manager.raw_data.describe())