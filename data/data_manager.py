import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import date, datetime, timedelta

ROOT_URL = 'https://www.nepalstock.com'

try:
    from wasmtime import Store, Module, Instance
    from retrying import retry
    import requests
    from urllib3 import disable_warnings
    disable_warnings()
except ImportError as e:
    raise ImportError(f"Missing dependencies: {e}")


WASM_FILE = os.path.join(os.path.dirname(__file__), '..', 'scraper', 'nepse_scraper', 'nepse.wasm')


class TokenParser:
    def __init__(self):
        self.store = Store()
        if not os.path.exists(WASM_FILE):
            raise FileNotFoundError(f"WASM file not found at: {WASM_FILE}")
        module = Module.from_file(self.store.engine, WASM_FILE)
        instance = Instance(self.store, module, [])
        self.cdx = instance.exports(self.store)["cdx"]
        self.rdx = instance.exports(self.store)["rdx"]
        self.bdx = instance.exports(self.store)["bdx"]
        self.ndx = instance.exports(self.store)["ndx"]
        self.mdx = instance.exports(self.store)["mdx"]

    def parse_token_response(self, token_response):
        n = self.cdx(self.store, token_response['salt1'], token_response['salt2'], token_response['salt3'], token_response['salt4'], token_response['salt5'])
        l = self.rdx(self.store, token_response['salt1'], token_response['salt2'], token_response['salt4'], token_response['salt3'], token_response['salt5'])
        o = self.bdx(self.store, token_response['salt1'], token_response['salt2'], token_response['salt4'], token_response['salt3'], token_response['salt5'])
        p = self.ndx(self.store, token_response['salt1'], token_response['salt2'], token_response['salt4'], token_response['salt3'], token_response['salt5'])
        q = self.mdx(self.store, token_response['salt1'], token_response['salt2'], token_response['salt4'], token_response['salt3'], token_response['salt5'])
        
        access_token = token_response['accessToken']
        refresh_token = token_response['refreshToken']
        
        parsed_access = access_token[0:n] + access_token[n+1:l] + access_token[l+1:o] + access_token[o+1:p] + access_token[p+1:q] + access_token[q+1:]
        
        return (parsed_access, token_response)


class NepseAPI:
    def __init__(self):
        self.token_parser = TokenParser()
        self.token_url = ROOT_URL + '/api/authenticate/prove'
        self.headers = {
            'Host': 'www.nepalstock.com',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
        }
        self._cached_token = None

    @retry(wait_fixed=2000, stop_max_attempt_number=5)
    def _get_valid_token(self):
        token_response = requests.get(self.token_url, headers=self.headers, verify=False).json()
        for salt_index in range(1, 6):
            token_response[f'salt{salt_index}'] = int(token_response[f'salt{salt_index}'])
        return self.token_parser.parse_token_response(token_response)

    def get_valid_token(self):
        if self._cached_token is None:
            self._cached_token = self._get_valid_token()
        return self._cached_token[0]

    def get_all_securities(self):
        token = self.get_valid_token()
        auth_headers = {'Authorization': f'Salter {token}', **self.headers}
        
        resp = requests.get(
            f'{ROOT_URL}/api/nots/security',
            headers=auth_headers,
            verify=False,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def get_price_history(self, ticker_id):
        token = self.get_valid_token()
        auth_headers = {'Authorization': f'Salter {token}', **self.headers}
        
        resp = requests.get(
            f'{ROOT_URL}/api/nots/market/graphdata/{ticker_id}',
            headers=auth_headers,
            verify=False,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()


class DataManager:
    def __init__(self, data_path='data/nepse_prices.csv'):
        self.data_path = data_path
        self.raw_data = None
        self.log_returns = None
        self.nepse = NepseAPI()

    def get_data(self, tickers=None, start_date=None, end_date=None, force_fetch=False):
        if not force_fetch and os.path.exists(self.data_path):
            print(f"Loading data from local file: {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path, index_col='date', parse_dates=True)
            self.raw_data.index = pd.to_datetime(self.raw_data.index).date
            print("Data loaded successfully from CSV.")
            return self.raw_data

        if tickers is None:
            print("Error: Ticker list must be provided to fetch new data.")
            return None

        print("Local data not found or `force_fetch` is True. Fetching from NEPSE...")
        
        try:
            securities = self.nepse.get_all_securities()
            symbol_to_id = {s['symbol']: s['id'] for s in securities}
        except Exception as e:
            print(f"Failed to get securities: {e}")
            print("Creating dummy data instead...")
            return self._create_dummy(tickers)

        return self._fetch_historical_data(tickers, symbol_to_id, start_date, end_date)

    def _fetch_historical_data(self, tickers, symbol_to_id, start_date, end_date):
        print(f"Fetching historical data for {len(tickers)} tickers...")
        
        all_data = []
        for i, ticker in enumerate(tickers):
            ticker = ticker.upper()
            if ticker not in symbol_to_id:
                print(f"Ticker {ticker} not found in securities")
                continue
            
            try:
                print(f"({i+1}/{len(tickers)}) Fetching {ticker}...")
                sec_id = symbol_to_id[ticker]
                history = self.nepse.get_price_history(sec_id)
                
                if history and len(history) > 0:
                    df = pd.DataFrame(history)
                    if 'x' in df.columns and 'y' in df.columns:
                        df['date'] = pd.to_datetime(df['x'], unit='ms').dt.date
                        df = df[['date', 'y']]
                        df.rename(columns={'y': ticker}, inplace=True)
                        all_data.append(df.set_index('date'))
                time.sleep(0.5)
            except Exception as e:
                print(f"Could not fetch data for {ticker}. Reason: {e}")

        if not all_data:
            print("No data was fetched. Creating dummy data instead...")
            return self._create_dummy(tickers)

        combined_df = pd.concat(all_data, axis=1)
        combined_df.ffill(inplace=True)
        combined_df.bfill(inplace=True)

        self.raw_data = combined_df
        print("Historical data fetched and cleaned successfully.")
        
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
    TARGET_TICKERS = ['NABIL', 'NICA', 'HDL', 'CIT', 'UPPER', 'GBIME', 'SCB', 'EBL']
    DATA_FILE = 'data/nepse_prices.csv'
    
    manager = DataManager(data_path=DATA_FILE)
    data = manager.get_data(tickers=TARGET_TICKERS, force_fetch=False)

    if data is not None:
        log_returns = manager.calculate_log_returns()
        if log_returns is not None:
            print("\n--- Calculated Log Returns (Sample) ---")
            print(log_returns.head())
            print("\n--- Data Description ---")
            print(manager.raw_data.describe())