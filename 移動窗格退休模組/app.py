import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
import pandas as pd
import multiprocessing as mp
from functools import partial

from MarketData import MarketData
from ResultAnalysis import ResultAnalysis
from RetirementSimulation import RetirementSimulation
from RollingWindowAnalysis import RollingWindowAnalysis

def main():
    # 1. 執行滾動時間窗口分析
    print("執行滾動時間窗口分析...")
    rolling_analysis = RollingWindowAnalysis(1970, 2023, 20, 10000)
    rolling_results = rolling_analysis.run_analysis()
    RollingWindowAnalysis.print_rolling_statistics(rolling_results)
    RollingWindowAnalysis.plot_rolling_results(rolling_results)

    # 2. 使用最新的數據進行單次模擬
    print("\n使用最新數據進行單次模擬...")
    historical_returns = MarketData.get_sp500_returns('2000-01-01', '2020-12-31')

    return_mean, return_std = MarketData.calculate_returns_volatility(historical_returns)
    print(f"歷史年化回報率: {return_mean:.2%}")
    print(f"歷史年化波動率: {return_std:.2%}")

    states, transition_matrix = MarketData.define_market_states(historical_returns)
    print("市場狀態參數:")
    for state, params in states.items():
        print(f"{state}: 平均回報率 = {params['mean']:.2%}, 標準差 = {params['std']:.2%}")
    print("轉換概率矩陣:")
    print(transition_matrix)

    params = {
        'initial_portfolio': 20000000,
        'years': 30,
        'withdrawal_rate': 0.04,
        'inflation_rate': 0.02,
        'initial_withdrawal': 400000,
        'additional_income': 200000,
        'emergency_fund': 1000000,
        'simulations': 10000
    }

    simulation = RetirementSimulation(params)
    results = simulation.monte_carlo_simulation(states, transition_matrix)

    success_rate, successes, failures, final_portfolios, median_final_portfolio, all_portfolios, min_withdrawals, max_withdrawals, avg_withdrawals, zero_withdrawal_percentage = results

    growth_rates = (all_portfolios[:, -1] / all_portfolios[:, 0]) ** (1/params['years']) - 1

    ResultAnalysis.print_statistics(success_rate, successes, failures, median_final_portfolio, final_portfolios, growth_rates, min_withdrawals, max_withdrawals, avg_withdrawals, zero_withdrawal_percentage, params['initial_withdrawal'], params['years'])

    ResultAnalysis.plot_results(params['initial_portfolio'],success_rate, final_portfolios, all_portfolios, min_withdrawals, max_withdrawals, avg_withdrawals, zero_withdrawal_percentage, params['years'], params['simulations'])

    # 3. 比較滾動分析和單次模擬結果
    print("\n比較滾動分析和單次模擬結果:")
    print(f"滾動分析平均成功率: {rolling_results['success_rate'].mean():.2%}")
    print(f"單次模擬成功率: {success_rate:.2%}")
    print(f"滾動分析平均中位數最終資產: {rolling_results['median_final_portfolio'].mean():,.0f} 台幣")
    print(f"單次模擬中位數最終資產: {median_final_portfolio:,.0f} 台幣")

if __name__ == "__main__":
    main()