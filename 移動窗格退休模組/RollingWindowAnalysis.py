import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
import pandas as pd
import multiprocessing as mp
from functools import partial

from MarketData import MarketData
from RetirementSimulation import RetirementSimulation
class RollingWindowAnalysis:
    """滾動窗口分析的類"""

    def __init__(self, start_year, end_year, window_size, simulations):
        self.start_year = start_year
        self.end_year = end_year
        self.window_size = window_size
        self.simulations = simulations

    def process_single_window(self, year):
        """處理單個時間窗口"""
        window_start = f"{year}-01-01"
        window_end = f"{year + self.window_size - 1}-12-31"
        historical_returns = MarketData.get_sp500_returns(window_start, window_end)
        
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

    def run_analysis(self):
        """執行滾動窗口分析"""
        years_range = range(self.start_year, self.end_year - self.window_size + 2)
        
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.process_single_window, years_range), 
                                total=len(years_range), desc="Processing rolling windows"))
        
        return pd.DataFrame(results)

    @staticmethod
    def plot_rolling_results(rolling_results):
        """繪製滾動分析結果圖表"""
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 成功率隨時間變化
        axs[0, 0].plot(rolling_results['window_start'], rolling_results['success_rate'])
        axs[0, 0].set_title('退休成功率隨時間變化')
        axs[0, 0].set_xlabel('起始年份')
        axs[0, 0].set_ylabel('成功率')

        # 中位數最終資產隨時間變化
        axs[0, 1].plot(rolling_results['window_start'], rolling_results['median_final_portfolio'])
        axs[0, 1].set_title('中位數最終資產隨時間變化')
        axs[0, 1].set_xlabel('起始年份')
        axs[0, 1].set_ylabel('資產價值 (台幣)')

        # 年化回報率隨時間變化
        axs[1, 0].plot(rolling_results['window_start'], rolling_results['return_mean'])
        axs[1, 0].set_title('年化回報率隨時間變化')
        axs[1, 0].set_xlabel('起始年份')
        axs[1, 0].set_ylabel('年化回報率')

        # 年化波動率隨時間變化
        axs[1, 1].plot(rolling_results['window_start'], rolling_results['return_std'])
        axs[1, 1].set_title('年化波動率隨時間變化')
        axs[1, 1].set_xlabel('起始年份')
        axs[1, 1].set_ylabel('年化波動率')

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
