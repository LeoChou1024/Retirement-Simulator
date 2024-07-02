import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
import pandas as pd
import multiprocessing as mp
from functools import partial

class MarketData:
    """處理市場數據的類"""

    @staticmethod
    def get_sp500_returns(start_date='1950-01-01', end_date='2023-12-31'):
        """獲取S&P 500的歷史年度回報率"""
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(start=start_date, end=end_date)
        annual_returns = hist['Close'].resample('Y').last().pct_change().dropna()
        return annual_returns

    @staticmethod
    def calculate_returns_volatility(returns):
        """計算歷史數據的年化回報率和波動率"""
        annual_return = returns.mean()
        annual_volatility = returns.std()
        return annual_return, annual_volatility

    @staticmethod
    def define_market_states(returns):
        """定義市場狀態並計算轉換概率矩陣"""
        bull_threshold = returns.mean() + 0.5 * returns.std()
        bear_threshold = returns.mean() - 0.5 * returns.std()

        states = {
            'bull': {'mean': returns[returns > bull_threshold].mean(), 'std': returns[returns > bull_threshold].std()},
            'bear': {'mean': returns[returns < bear_threshold].mean(), 'std': returns[returns < bear_threshold].std()},
            'neutral': {'mean': returns[(returns >= bear_threshold) & (returns <= bull_threshold)].mean(), 
                        'std': returns[(returns >= bear_threshold) & (returns <= bull_threshold)].std()}
        }

        state_sequence = np.where(returns > bull_threshold, 0, np.where(returns < bear_threshold, 1, 2))
        transition_matrix = np.zeros((3, 3))

        for i in range(len(state_sequence) - 1):
            transition_matrix[state_sequence[i], state_sequence[i+1]] += 1

        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

        return states, transition_matrix