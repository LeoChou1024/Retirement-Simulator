import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
import pandas as pd

def get_sp500_returns(start_date='1950-01-01', end_date='2023-12-31'):
    """
    獲取S&P 500的歷史年度回報率
    :param start_date: 開始日期
    :param end_date: 結束日期
    :return: 年度回報率DataFrame
    """
    sp500 = yf.Ticker("^GSPC")
    hist = sp500.history(start=start_date, end=end_date)
    annual_returns = hist['Close'].resample('Y').last().pct_change().dropna()
    return annual_returns

def calculate_returns_volatility(returns):
    """
    計算歷史數據的年化回報率和波動率。
    """
    annual_return = returns.mean()
    annual_volatility = returns.std()
    return annual_return, annual_volatility

def define_market_states(returns):
    """
    定義市場狀態並計算轉換概率矩陣。
    """
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

def classify_market_state(return_rate, states):
    """
    根據回報率分類市場狀態
    """
    if return_rate > states['bull']['mean']:
        return "bull"
    elif return_rate < states['bear']['mean']:
        return "bear"
    else:
        return "neutral"

def generate_market_cycle(years, historical_returns, states, transition_matrix):
    """
    根據歷史數據和馬爾可夫鏈生成市場週期
    """
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

def monte_carlo_retirement_simulation(
    initial_portfolio, years, withdrawal_rate, inflation_rate,
    states, transition_matrix, simulations, initial_withdrawal,
    additional_income, emergency_fund
):
    results = []
    final_portfolios = []
    all_portfolios = np.zeros((simulations, years+1))
    all_portfolios[:, 0] = initial_portfolio
    all_withdrawals = np.zeros((simulations, years))
    zero_withdrawal_years = np.zeros((simulations, years), dtype=bool)
    
    for sim in range(simulations):
        portfolio = initial_portfolio
        market_returns, market_states = generate_market_cycle(years, None, states, transition_matrix)
        
        for year in range(years):
            current_withdrawal = initial_withdrawal * (1 + inflation_rate)**year
            max_withdrawal = min(initial_withdrawal * 3 * (1 + inflation_rate)**year, 1200000 * (1 + inflation_rate)**year)
            
            if market_states[year] == 'bull':
                withdrawal = min(max(current_withdrawal, portfolio * withdrawal_rate), max_withdrawal)
            elif market_states[year] == 'bear':
                withdrawal = min(current_withdrawal, portfolio * 0.03)
            else:  # neutral
                withdrawal = min(max(current_withdrawal, portfolio * (withdrawal_rate * 0.75)), max_withdrawal)
            
            if portfolio <= emergency_fund:
                withdrawal = 0
                zero_withdrawal_years[sim, year] = True
            
            all_withdrawals[sim, year] = withdrawal
            portfolio -= withdrawal
            portfolio += additional_income
            
            annual_return = market_returns[year]
            portfolio *= (1 + annual_return)
            
            all_portfolios[sim, year+1] = portfolio
            
            if portfolio <= 0:
                results.append(False)
                final_portfolios.append(0)
                all_portfolios[sim, year+1:] = 0
                zero_withdrawal_years[sim, year+1:] = True
                break
            
            if year == years - 1:
                results.append(True)
                final_portfolios.append(portfolio)
    
    success_count = sum(results)
    success_rate = success_count / len(results)
    median_final_portfolio = np.median(final_portfolios)
    
    min_withdrawals = np.min(all_withdrawals, axis=0)
    max_withdrawals = np.max(all_withdrawals, axis=0)
    avg_withdrawals = np.mean(all_withdrawals, axis=0)
    zero_withdrawal_percentage = np.mean(zero_withdrawal_years, axis=0) * 100
    
    return success_rate, success_count, simulations - success_count, final_portfolios, median_final_portfolio, all_portfolios, min_withdrawals, max_withdrawals, avg_withdrawals, zero_withdrawal_percentage

def plot_results(success_rate, final_portfolios, all_portfolios, min_withdrawals, max_withdrawals, avg_withdrawals, zero_withdrawal_percentage, years, simulations):
    """
    繪製模擬結果的圖表。

    參數:
    success_rate (float): 退休計劃成功率
    final_portfolios (list): 最終投資組合價值列表
    all_portfolios (numpy.array): 所有模擬的投資組合價值
    min_withdrawals (numpy.array): 最小提款金額
    max_withdrawals (numpy.array): 最大提款金額
    years (int): 模擬年數
    simulations (int): 模擬次數
    """
    fig = plt.figure(figsize=(20, 25))  # 增加图表的高度
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 退休計劃成功率柱狀圖
    ax1 = fig.add_subplot(4, 3, 1)
    ax1.bar(['成功', '失敗'], [success_rate, 1-success_rate], color=['green', 'red'])
    ax1.set_title('退休計劃成功率')
    ax1.set_ylabel('概率')
    ax1.set_ylim(0, 1)

    # 2. 最終資產分佈直方圖
    ax2 = fig.add_subplot(4, 3, 2)
    ax2.hist(final_portfolios, bins=50, edgecolor='black')
    ax2.set_title('最終資產分佈')
    ax2.set_xlabel('資產價值 (台幣)')
    ax2.set_ylabel('頻率')

    # 3. 最終資產箱形圖
    ax3 = fig.add_subplot(4, 3, 3)
    ax3.boxplot(final_portfolios)
    ax3.set_title('最終資產箱形圖')
    ax3.set_ylabel('資產價值 (台幣)')

    # 4. 最終資產累積分佈函數
    ax4 = fig.add_subplot(4, 3, 4)
    ax4.hist(final_portfolios, bins=50, cumulative=True, density=True, histtype='step')
    ax4.set_title('最終資產累積分佈函數')
    ax4.set_xlabel('資產價值 (台幣)')
    ax4.set_ylabel('累積概率')

    # 5. 年度資產變化
    ax5 = fig.add_subplot(4, 3, 5)
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        ax5.plot(np.percentile(all_portfolios, p, axis=0), label=f'{p}th percentile')
    ax5.set_title('年度資產變化')
    ax5.set_xlabel('年')
    ax5.set_ylabel('資產價值 (台幣)')
    ax5.legend()

    # 6. 資產成長率分佈
    ax6 = fig.add_subplot(4, 3, 6)
    growth_rates = (all_portfolios[:, -1] / all_portfolios[:, 0]) ** (1/years) - 1
    ax6.hist(growth_rates, bins=50, edgecolor='black')
    ax6.set_title('年化資產成長率分佈')
    ax6.set_xlabel('年化成長率')
    ax6.set_ylabel('頻率')

    # 7. 風險值 (Value at Risk) 分析
    ax7 = fig.add_subplot(4, 3, 7)
    var_95 = initial_portfolio - np.percentile(final_portfolios, 5)
    var_99 = initial_portfolio - np.percentile(final_portfolios, 1)
    ax7.hist(final_portfolios, bins=50, edgecolor='black')
    ax7.axvline(var_95, color='r', linestyle='dashed', label='95% VaR')
    ax7.axvline(var_99, color='g', linestyle='dashed', label='99% VaR')
    ax7.set_title('風險值 (VaR) 分析')
    ax7.set_xlabel('最終資產價值 (台幣)')
    ax7.set_ylabel('頻率')
    ax7.legend()

    # 8. 資產軌跡
    ax8 = fig.add_subplot(4, 3, 8)
    for i in range(min(100, simulations)):
        ax8.plot(all_portfolios[i, :], alpha=0.1)
    ax8.set_title('資產軌跡 (100次模擬)')
    ax8.set_xlabel('年')
    ax8.set_ylabel('資產價值 (台幣)')

    # 9. 提款金額變化
    ax9 = fig.add_subplot(4, 3, 9)
    years_range = range(years)
    ax9.plot(years_range, min_withdrawals, label='最小提款')
    ax9.plot(years_range, max_withdrawals, label='最大提款')
    ax9.set_title('年度提款金額範圍')
    ax9.set_xlabel('年')
    ax9.set_ylabel('提款金額 (台幣)')
    ax9.legend()

    # 10. 平均提款金额变化
    ax10 = fig.add_subplot(4, 3, 10)
    ax10.plot(range(years), avg_withdrawals)
    ax10.set_title('年度平均提款金额')
    ax10.set_xlabel('年')
    ax10.set_ylabel('提款金额 (台币)')

    # 11. 零提款年份百分比
    ax11 = fig.add_subplot(4, 3, 11)
    ax11.bar(range(years), zero_withdrawal_percentage)
    ax11.set_title('零提款年份百分比')
    ax11.set_xlabel('年')
    ax11.set_ylabel('百分比')
    ax11.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()

def print_statistics(success_rate, successes, failures, median_final_portfolio, final_portfolios, growth_rates, min_withdrawals, max_withdrawals, avg_withdrawals, zero_withdrawal_percentage, initial_withdrawal, years):
    """
    打印模擬結果的統計信息。

    參數:
    [參數說明與之前的函數相似]
    """
    print(f"退休成功率: {success_rate:.2%}")
    print(f"成功次數: {successes}")
    print(f"失敗次數: {failures}")
    print(f"資產中位數: {median_final_portfolio:,.0f} 台幣")

    final_percentiles = np.percentile(final_portfolios, [10, 25, 50, 75, 90])
    growth_rates_percentiles = np.percentile(growth_rates, [0, 25, 50, 75, 100])
    max_final_asset = np.max(final_portfolios)
    min_final_asset = np.min(final_portfolios)
    positive_final_assets = np.sum(np.array(final_portfolios) > 0)

    print(f"\n詳細統計信息:")
    print(f"10th percentile 最終資產: {final_percentiles[0]:,.0f} 台幣")
    print(f"25th percentile 最終資產: {final_percentiles[1]:,.0f} 台幣")
    print(f"50th percentile 最終資產: {final_percentiles[2]:,.0f} 台幣")
    print(f"75th percentile 最終資產: {final_percentiles[3]:,.0f} 台幣")
    print(f"90th percentile 最終資產: {final_percentiles[4]:,.0f} 台幣")

    print(f"\n最小年化成長率: {growth_rates_percentiles[0]:.2%}")
    print(f"25th percentile 年化成長率: {growth_rates_percentiles[1]:.2%}")
    print(f"中位數年化成長率: {growth_rates_percentiles[2]:.2%}")
    print(f"75th percentile 年化成長率: {growth_rates_percentiles[3]:.2%}")
    print(f"最大年化成長率: {growth_rates_percentiles[4]:.2%}")

    print(f"\n最終資產最大值: {max_final_asset:,.0f} 台幣")
    print(f"最終資產最小值: {min_final_asset:,.0f} 台幣")
    print(f"{years}年後仍有正資產的模擬次數: {positive_final_assets}")

    print(f"\n初始提款: {initial_withdrawal:,.0f} 台币")
    print(f"{years}年后预估最小提款: {min_withdrawals[-1]:,.0f} 台币")
    print(f"{years}年后预估最大提款: {max_withdrawals[-1]:,.0f} 台币")
    print(f"{years}年后预估平均提款: {avg_withdrawals[-1]:,.0f} 台币")
    print(f"零提款年份平均百分比: {np.mean(zero_withdrawal_percentage):.2f}%")
    print(f"最高零提款百分比: {np.max(zero_withdrawal_percentage):.2f}%")

    var_95 = np.percentile(final_portfolios, 5)
    var_99 = np.percentile(final_portfolios, 1)
    print(f"\n年化資產成長率中位數: {np.median(growth_rates):.2%}")
    print(f"95% 風險值 (VaR): {var_95:,.0f} 台幣")
    print(f"99% 風險值 (VaR): {var_99:,.0f} 台幣")

def rolling_window_analysis(start_year, end_year, window_size=20, simulations=10000):
    results = []
    years_range = range(start_year, end_year - window_size + 2)
    
    for year in tqdm(years_range, desc="Processing rolling windows"):
        window_start = f"{year}-01-01"
        window_end = f"{year + window_size - 1}-12-31"
        historical_returns = get_sp500_returns(window_start, window_end)
        
        return_mean, return_std = calculate_returns_volatility(historical_returns)
        states, transition_matrix = define_market_states(historical_returns)
        
        simulation_result = monte_carlo_retirement_simulation(
            initial_portfolio=20000000,
            years=30,
            withdrawal_rate=0.04,
            inflation_rate=0.02,
            states=states,
            transition_matrix=transition_matrix,
            simulations=simulations,
            initial_withdrawal=400000,
            additional_income=200000, 
            emergency_fund=1000000
        )
        
        success_rate, _, _, final_portfolios, median_final_portfolio, _, _, _, _, _ = simulation_result
        
        results.append({
            'window_start': year,
            'window_end': year + window_size - 1,
            'success_rate': success_rate,
            'median_final_portfolio': median_final_portfolio,
            'return_mean': return_mean,
            'return_std': return_std
        })
    
    return pd.DataFrame(results)

def plot_rolling_results(rolling_results):
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

def print_rolling_statistics(rolling_results):
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


if __name__ == "__main__":
    # 1. 執行滾動時間窗口分析
    print("執行滾動時間窗口分析...")
    rolling_results = rolling_window_analysis(1970, 2023,20)
    # 打印滾動分析統計結果
    print_rolling_statistics(rolling_results)

    # 繪製滾動分析結果圖表
    plot_rolling_results(rolling_results)

    # 2. 使用最新的數據進行單次模擬
    print("\n使用最新數據進行單次模擬...")
    historical_returns = get_sp500_returns('2000-01-01', '2020-12-31')

    # 計算歷史回報率和波動率
    return_mean, return_std = calculate_returns_volatility(historical_returns)
    print(f"歷史年化回報率: {return_mean:.2%}")
    print(f"歷史年化波動率: {return_std:.2%}")

    # 定義市場狀態和轉換概率
    states, transition_matrix = define_market_states(historical_returns)
    print("市場狀態參數:")
    for state, params in states.items():
        print(f"{state}: 平均回報率 = {params['mean']:.2%}, 標準差 = {params['std']:.2%}")
    print("轉換概率矩陣:")
    print(transition_matrix)

    # 設定模擬參數
    initial_portfolio = 20000000
    years = 30
    withdrawal_rate = 0.04
    inflation_rate = 0.02
    simulations = 10000
    initial_withdrawal = 400000
    additional_income = 200000
    emergency_fund = 1000000

    # 執行模擬
    results = monte_carlo_retirement_simulation(
        initial_portfolio, years, withdrawal_rate, inflation_rate,
        states, transition_matrix, simulations, initial_withdrawal,
        additional_income, emergency_fund
    )

    success_rate, successes, failures, final_portfolios, median_final_portfolio, all_portfolios, min_withdrawals, max_withdrawals, avg_withdrawals, zero_withdrawal_percentage = results

    # 計算成長率
    growth_rates = (all_portfolios[:, -1] / all_portfolios[:, 0]) ** (1/years) - 1

    # 打印統計信息
    print_statistics(success_rate, successes, failures, median_final_portfolio, final_portfolios, growth_rates, min_withdrawals, max_withdrawals, avg_withdrawals, zero_withdrawal_percentage, initial_withdrawal, years)

    # 繪製圖表
    plot_results(success_rate, final_portfolios, all_portfolios, min_withdrawals, max_withdrawals, avg_withdrawals, zero_withdrawal_percentage, years, simulations)

    # 3. 比較滾動分析和單次模擬結果
    print("\n比較滾動分析和單次模擬結果:")
    print(f"滾動分析平均成功率: {rolling_results['success_rate'].mean():.2%}")
    print(f"單次模擬成功率: {success_rate:.2%}")
    print(f"滾動分析平均中位數最終資產: {rolling_results['median_final_portfolio'].mean():,.0f} 台幣")
    print(f"單次模擬中位數最終資產: {median_final_portfolio:,.0f} 台幣")
