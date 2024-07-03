import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

from MarketData import MarketData
from RetirementSimulation import RetirementSimulation

class RollingWindowAnalysis:
    def __init__(self, start_year, end_year, window_size, simulations):
        self.start_year = start_year
        self.end_year = end_year
        self.window_size = window_size
        self.simulations = simulations

    def process_single_window(self, year, sp500_weight, nasdaq_weight):
        """處理單個時間窗口"""
        window_start = f"{year}-01-01"
        window_end = f"{year + self.window_size - 1}-12-31"
        historical_returns = MarketData.get_combined_returns(sp500_weight, nasdaq_weight, window_start, window_end)
        
        if historical_returns.empty:
            return {
                'window_start': year,
                'window_end': year + self.window_size - 1,
                'success_rate': np.nan,
                'median_final_portfolio': np.nan,
                'return_mean': np.nan,
                'return_std': np.nan
            }

        return_mean, return_std = MarketData.calculate_returns_volatility(historical_returns)
        states, transition_matrix = MarketData.define_market_states(historical_returns)
        
        params = {
            'initial_portfolio': 20000000,
            'years': 30,
            'withdrawal_rate': 0.04,
            'inflation_rate': 0.02,
            'initial_withdrawal': 400000,
            'additional_income': 200000,
            'emergency_fund': 1000000,
            'simulations': self.simulations
        }
        
        simulation = RetirementSimulation(params)
        simulation_result = simulation.monte_carlo_simulation(states, transition_matrix)
        
        success_rate, _, _, _, median_final_portfolio, _, _, _, _, _ = simulation_result
        
        return {
            'window_start': year,
            'window_end': year + self.window_size - 1,
            'success_rate': success_rate,
            'median_final_portfolio': median_final_portfolio,
            'return_mean': return_mean,
            'return_std': return_std
        }

    def _process_window_wrapper(self, args):
        """包裝函數，用於多進程處理"""
        return self.process_single_window(*args)

    def run_analysis_for_portfolio(self, sp500_weight, nasdaq_weight):
        """執行特定投資組合配置的滾動窗口分析"""
        years_range = range(self.start_year, self.end_year - self.window_size + 2)
        args_list = [(year, sp500_weight, nasdaq_weight) for year in years_range]
        
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(self._process_window_wrapper, args_list), 
                                total=len(years_range), desc=f"處理滾動窗口 (S&P 500: {sp500_weight*100}%, 納斯達克: {nasdaq_weight*100}%)"))
        
        return pd.DataFrame(results)

    def run_analysis_for_multiple_portfolios(self, portfolio_weights):
        """執行多個投資組合配置的滾動窗口分析"""
        results = {}
        for sp500_weight, nasdaq_weight in portfolio_weights:
            key = f"S&P500_{sp500_weight*100:g}%_納斯達克_{nasdaq_weight*100:g}%"
            results[key] = self.run_analysis_for_portfolio(sp500_weight, nasdaq_weight)
        return results

    @staticmethod
    def plot_rolling_results(rolling_results):
        """繪製滾動分析結果圖表"""
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        axs[0, 0].plot(rolling_results['window_start'], rolling_results['success_rate'])
        axs[0, 0].set_title('退休成功率隨時間變化')
        axs[0, 0].set_xlabel('起始年份')
        axs[0, 0].set_ylabel('成功率')

        axs[0, 1].plot(rolling_results['window_start'], rolling_results['median_final_portfolio'])
        axs[0, 1].set_title('中位數最終資產隨時間變化')
        axs[0, 1].set_xlabel('起始年份')
        axs[0, 1].set_ylabel('資產價值 (台幣)')

        axs[1, 0].plot(rolling_results['window_start'], rolling_results['return_mean'])
        axs[1, 0].set_title('年化回報率隨時間變化')
        axs[1, 0].set_xlabel('起始年份')
        axs[1, 0].set_ylabel('年化回報率')

        axs[1, 1].plot(rolling_results['window_start'], rolling_results['return_std'])
        axs[1, 1].set_title('年化波動率隨時間變化')
        axs[1, 1].set_xlabel('起始年份')
        axs[1, 1].set_ylabel('年化波動率')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multiple_portfolio_results(results):
        """繪製多個投資組合的滾動分析結果比較圖表"""
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        for portfolio, data in results.items():
            axs[0, 0].plot(data['window_start'], data['success_rate'], label=portfolio)
            axs[0, 1].plot(data['window_start'], data['median_final_portfolio'], label=portfolio)
            axs[1, 0].plot(data['window_start'], data['return_mean'], label=portfolio)
            axs[1, 1].plot(data['window_start'], data['return_std'], label=portfolio)

        axs[0, 0].set_title('退休成功率隨時間變化')
        axs[0, 0].set_xlabel('起始年份')
        axs[0, 0].set_ylabel('成功率')
        axs[0, 0].legend()

        axs[0, 1].set_title('中位數最終資產隨時間變化')
        axs[0, 1].set_xlabel('起始年份')
        axs[0, 1].set_ylabel('資產價值 (台幣)')
        axs[0, 1].legend()

        axs[1, 0].set_title('年化回報率隨時間變化')
        axs[1, 0].set_xlabel('起始年份')
        axs[1, 0].set_ylabel('年化回報率')
        axs[1, 0].legend()

        axs[1, 1].set_title('年化波動率隨時間變化')
        axs[1, 1].set_xlabel('起始年份')
        axs[1, 1].set_ylabel('年化波動率')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_rolling_statistics(rolling_results):
        """打印滾動分析統計結果"""
        print("\n滾動時間窗口分析統計結果:")
        print(f"平均成功率: {rolling_results['success_rate'].mean():.2%}")
        print(f"成功率中位數: {rolling_results['success_rate'].median():.2%}")
        print(f"成功率標準差: {rolling_results['success_rate'].std():.2%}")
        print(f"\n平均中位數最終資產: {rolling_results['median_final_portfolio'].mean():,.0f} 台幣")
        print(f"中位數最終資產的中位數: {rolling_results['median_final_portfolio'].median():,.0f} 台幣")
        print(f"中位數最終資產標準差: {rolling_results['median_final_portfolio'].std():,.0f} 台幣")
        print(f"\n平均年化回報率: {rolling_results['return_mean'].mean():.2%}")
        print(f"年化回報率中位數: {rolling_results['return_mean'].median():.2%}")
        print(f"年化回報率標準差: {rolling_results['return_mean'].std():.2%}")
        print(f"\n平均年化波動率: {rolling_results['return_std'].mean():.2%}")
        print(f"年化波動率中位數: {rolling_results['return_std'].median():.2%}")
        print(f"年化波動率標準差: {rolling_results['return_std'].std():.2%}")