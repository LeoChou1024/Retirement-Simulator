import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
import pandas as pd
import multiprocessing as mp
from functools import partial
class RetirementSimulation:
    """退休模擬的類"""

    def __init__(self, params):
        """初始化模擬參數"""
        self.params = params

    def generate_market_cycle(self, years, states, transition_matrix):
        """根據歷史數據和馬爾可夫鏈生成市場週期"""
        current_state = np.random.choice(3)
        cycle = []
        states_cycle = []
        state_names = ['bull', 'bear', 'neutral']

        for _ in range(years):
            state_name = state_names[current_state]
            annual_return = np.random.normal(states[state_name]['mean'], states[state_name]['std'])
            cycle.append(annual_return)
            states_cycle.append(state_name)
            current_state = np.random.choice(3, p=transition_matrix[current_state])

        return cycle, states_cycle

    def monte_carlo_simulation(self, states, transition_matrix):
        """執行蒙特卡羅模擬"""
        results = []
        final_portfolios = []
        all_portfolios = np.zeros((self.params['simulations'], self.params['years']+1))
        all_portfolios[:, 0] = self.params['initial_portfolio']
        all_withdrawals = np.zeros((self.params['simulations'], self.params['years']))
        zero_withdrawal_years = np.zeros((self.params['simulations'], self.params['years']), dtype=bool)
        
        for sim in range(self.params['simulations']):
            portfolio = self.params['initial_portfolio']
            market_returns, market_states = self.generate_market_cycle(self.params['years'], states, transition_matrix)
            
            for year in range(self.params['years']):
                current_withdrawal = self.params['initial_withdrawal'] * (1 + self.params['inflation_rate'])**year
                max_withdrawal = min(self.params['initial_withdrawal'] * 3 * (1 + self.params['inflation_rate'])**year, 
                                     1200000 * (1 + self.params['inflation_rate'])**year)
                
                if market_states[year] == 'bull':
                    withdrawal = min(max(current_withdrawal, portfolio * self.params['withdrawal_rate']), max_withdrawal)
                elif market_states[year] == 'bear':
                    withdrawal = min(current_withdrawal, portfolio * 0.03)
                else:  # neutral
                    withdrawal = min(max(current_withdrawal, portfolio * (self.params['withdrawal_rate'] * 0.75)), max_withdrawal)
                
                if portfolio <= self.params['emergency_fund']:
                    withdrawal = 0
                    zero_withdrawal_years[sim, year] = True
                
                all_withdrawals[sim, year] = withdrawal
                portfolio -= withdrawal
                portfolio += self.params['additional_income']
                
                annual_return = market_returns[year]
                portfolio *= (1 + annual_return)
                
                all_portfolios[sim, year+1] = portfolio
                
                if portfolio <= 0:
                    results.append(False)
                    final_portfolios.append(0)
                    all_portfolios[sim, year+1:] = 0
                    zero_withdrawal_years[sim, year+1:] = True
                    break
                
                if year == self.params['years'] - 1:
                    results.append(True)
                    final_portfolios.append(portfolio)
        
        success_count = sum(results)
        success_rate = success_count / len(results)
        median_final_portfolio = np.median(final_portfolios)
        
        min_withdrawals = np.min(all_withdrawals, axis=0)
        max_withdrawals = np.max(all_withdrawals, axis=0)
        avg_withdrawals = np.mean(all_withdrawals, axis=0)
        zero_withdrawal_percentage = np.mean(zero_withdrawal_years, axis=0) * 100
        
        return (success_rate, success_count, self.params['simulations'] - success_count, final_portfolios, 
                median_final_portfolio, all_portfolios, min_withdrawals, max_withdrawals, 
                avg_withdrawals, zero_withdrawal_percentage)