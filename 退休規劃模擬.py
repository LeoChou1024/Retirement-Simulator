import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# 獲取S&P 500的歷史數據
def get_sp500_returns(start_date='1950-01-01', end_date='2023-12-31'):
    """
    獲取S&P 500的歷史年度回報率
    :param start_date: 開始日期
    :param end_date: 結束日期
    :return: 年度回報率列表
    """
    sp500 = yf.Ticker("^GSPC")
    hist = sp500.history(start=start_date, end=end_date)
    annual_returns = hist['Close'].resample('Y').last().pct_change().dropna()
    return annual_returns.tolist()

# 生成市場週期
def generate_market_cycle(years, historical_returns):
    """
    根據歷史數據生成市場週期
    :param years: 模擬年數
    :param historical_returns: 歷史回報率
    :return: 生成的市場週期回報率
    """
    return np.random.choice(historical_returns, size=years)

# 模擬極端事件
def simulate_extreme_event(probability=0.2, severity=(-0.5, 0.1)):
    """
    模擬極端市場事件
    :param probability: 發生概率
    :param severity: 嚴重程度範圍 (最小值, 最大值)
    :return: 極端事件的回報率
    """
    if np.random.random() < probability:
        return np.random.uniform(severity[0], severity[1])
    return 0

# 動態提款策略
def dynamic_withdrawal(portfolio, previous_withdrawal, inflation_rate, return_rate, max_withdrawal=1200000):
    """
    動態調整提款金額
    :param portfolio: 當前投資組合價值
    :param previous_withdrawal: 上一年的提款金額
    :param inflation_rate: 通貨膨脹率
    :param return_rate: 投資回報率
    :param max_withdrawal: 最大提款金額
    :return: 本年度的提款金額
    """
    if return_rate > inflation_rate:
        withdrawal = min(previous_withdrawal * (1 + inflation_rate), portfolio * 0.05)
    else:
        withdrawal = max(previous_withdrawal * 0.95, portfolio * 0.03)
    return min(withdrawal, max_withdrawal)  # 確保不超過最大提款金額

# 計算稅款
def calculate_tax(income, capital_gains):
    """
    計算稅款
    :param income: 收入
    :param capital_gains: 資本利得
    :return: 總稅款
    """
    income_tax = income * 0.05  # 簡化的所得稅計算
    capital_gains_tax = capital_gains * 0.15  # 資本利得稅
    return 0
    #return income_tax + capital_gains_tax

# 主要的蒙特卡羅模擬函數
def monte_carlo_retirement(
    initial_portfolio=17500000,
    years=50,
    initial_withdrawal_rate=0.035,
    inflation_rate_mean=0.02,
    inflation_rate_std=0.01,
    stock_percentage_initial=1,
    bond_percentage_initial=0,
    simulations=10000,
    additional_income=0,
    medical_expenses_base=10000,
    medical_expenses_growth=0.08,
    max_withdrawal=1500000
):
    """
    進行蒙特卡羅退休模擬
    :param initial_portfolio: 初始投資組合價值
    :param years: 模擬年數
    :param initial_withdrawal_rate: 初始提款率
    :param inflation_rate_mean: 平均通貨膨脹率
    :param inflation_rate_std: 通貨膨脹率標準差
    :param stock_percentage_initial: 初始股票比例
    :param bond_percentage_initial: 初始債券比例
    :param simulations: 模擬次數
    :param additional_income: 額外收入
    :param medical_expenses_base: 基礎醫療支出
    :param medical_expenses_growth: 醫療支出年增長率
    :param max_withdrawal: 最大年度提款金額
    :return: 模擬結果和成功率
    """
    results = []
    historical_returns = get_sp500_returns()  # 獲取歷史回報率
    
    for _ in range(simulations):
        portfolio = initial_portfolio
        stock_percentage = stock_percentage_initial
        bond_percentage = bond_percentage_initial
        
        stock_returns = generate_market_cycle(years, historical_returns)
        bond_returns = np.random.normal(0.03, 0.05, years)
        
        previous_withdrawal = initial_portfolio * initial_withdrawal_rate
        
        for year in range(years):
            # 生成當年的通膨率
            inflation_rate = np.random.normal(inflation_rate_mean, inflation_rate_std)
            
            # 計算投資回報
            annual_return = (stock_returns[year] + simulate_extreme_event()) * stock_percentage + bond_returns[year] * bond_percentage
            
            # 計算提款金額
            withdrawal = dynamic_withdrawal(portfolio, previous_withdrawal, inflation_rate, annual_return, max_withdrawal)
            
            # 計算醫療支出
            medical_expenses = medical_expenses_base * (1 + medical_expenses_growth) ** year
            
            # 計算總收入和資本利得
            total_income = withdrawal + additional_income
            capital_gains = max(0, portfolio * annual_return)
            
            # 計算稅款
            tax = calculate_tax(total_income, capital_gains)
            
            # 更新投資組合價值
            portfolio = portfolio * (1 + annual_return) - withdrawal - medical_expenses - tax + additional_income
            
            if portfolio <= 0:
                results.append(0)
                break
            
            # 調整資產配置
            stock_percentage = max(0.4, stock_percentage_initial - 0.005 * year)
            bond_percentage = 1 - stock_percentage
            
            previous_withdrawal = withdrawal
        
        if portfolio > 0:
            results.append(portfolio)
    
    success_rate = len([r for r in results if r > 0]) / simulations * 100
    return results, success_rate

# 運行模擬並繪製圖表
def run_simulations_and_plot():
    scenarios = [
        ("100% 股票 0% 債券", 1, 0),
        ("80% 股票 20% 債券", 0.8, 0.2),
        ("70% 股票 30% 債券", 0.7, 0.3),
        ("60% 股票 40% 債券", 0.6, 0.4),
        ("50% 股票 50% 債券", 0.5, 0.5),
    ]

    plt.figure(figsize=(15, 10))
    results_table = []

    for scenario, stock_percentage, bond_percentage in scenarios:
        results, success_rate = monte_carlo_retirement(stock_percentage_initial=stock_percentage, bond_percentage_initial=bond_percentage)
        
        plt.hist(results, bins=50, alpha=0.5, label=f'{scenario} (成功率: {success_rate:.2f}%)')
        
        median_portfolio = np.median(results)
        mean_portfolio = np.mean(results)
        
        results_table.append({
            "情境": scenario,
            "成功率": f"{success_rate:.2f}%",
            "最終投資組合中位數": f"{median_portfolio:,.0f}",
            "最終投資組合平均值": f"{mean_portfolio:,.0f}"
        })
    print(pd.DataFrame(results_table))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode Ms'] # 修改中文字體
    plt.rcParams['axes.unicode_minus'] = False # 顯示負號
    plt.title('不同資產配置策略的退休投資組合價值分佈（40年模擬）')
    plt.xlabel('40年後的投資組合價值')
    plt.ylabel('頻率')
    plt.legend()
    plt.show()


# 運行模擬
run_simulations_and_plot()