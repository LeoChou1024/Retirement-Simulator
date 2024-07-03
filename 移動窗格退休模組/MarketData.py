import yfinance as yf
import pandas as pd
import numpy as np

class MarketData:
    @staticmethod
    def get_index_returns(ticker, start_date='1950-01-01', end_date='2023-12-31'):
        """獲取指定指數的歷史年度回報率"""
        index = yf.Ticker(ticker)
        try:
            hist = index.history(start=start_date, end=end_date)
            if hist.empty:
                print(f"警告：{ticker}在{start_date}和{end_date}之間沒有可用數據")
                return pd.Series(dtype=float)
            if not isinstance(hist.index, pd.DatetimeIndex):
                hist.index = pd.to_datetime(hist.index)
            annual_returns = hist['Close'].resample('Y').last().pct_change().dropna()
            return annual_returns
        except Exception as e:
            print(f"獲取{ticker}數據時出錯：{str(e)}")
            return pd.Series(dtype=float)

    @staticmethod
    def get_combined_returns(sp500_weight, nasdaq_weight, start_date='1950-01-01', end_date='2023-12-31'):
        """獲取 S&P 500 和納斯達克指數組合的歷史年度回報率"""
        sp500_returns = MarketData.get_index_returns("^GSPC", start_date, end_date)
        nasdaq_returns = MarketData.get_index_returns("^IXIC", start_date, end_date)

        if sp500_returns.empty and nasdaq_returns.empty:
            print(f"警告：S&P 500和納斯達克指數在{start_date}和{end_date}之間都沒有可用數據")
            return pd.Series(dtype=float)

        # 確保兩個序列有相同的日期索引
        common_index = sp500_returns.index.intersection(nasdaq_returns.index)
        if common_index.empty:
            print(f"警告：S&P 500和納斯達克指數在{start_date}和{end_date}之間沒有共同日期")
            return pd.Series(dtype=float)

        sp500_returns = sp500_returns[common_index]
        nasdaq_returns = nasdaq_returns[common_index]

        combined_returns = sp500_weight * sp500_returns + nasdaq_weight * nasdaq_returns
        return combined_returns

    @staticmethod
    def calculate_returns_volatility(returns):
        """計算歷史數據的年化回報率和波動率"""
        if returns.empty:
            return 0, 0
        annual_return = returns.mean()
        annual_volatility = returns.std()
        return annual_return, annual_volatility

    @staticmethod
    def define_market_states(returns):
        """定義市場狀態並計算轉換概率矩陣"""
        if returns.empty:
            return {'bull': {'mean': 0, 'std': 0}, 'bear': {'mean': 0, 'std': 0}, 'neutral': {'mean': 0, 'std': 0}}, np.zeros((3, 3))

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

        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = np.divide(transition_matrix, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)

        return states, transition_matrix