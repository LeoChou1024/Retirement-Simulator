import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def monte_carlo_retirement_simulation(
    initial_portfolio=20000000,
    years=50,
    withdrawal_rate=0.04,
    inflation_rate=0.02,
    return_mean=0.07,
    return_std=0.15,
    simulations=10000,
    initial_withdrawal=400000,
    additional_income=200000,
    emergency_fund=1000000
):
    results = []
    final_portfolios = []
    all_portfolios = np.zeros((simulations, years+1))
    all_portfolios[:, 0] = initial_portfolio
    all_withdrawals = np.zeros((simulations, years))
    
    # 使用對數正態分佈來生成回報率
    mu = np.log(1 + return_mean) - 0.5 * np.log(1 + return_std**2 / (1 + return_mean)**2)
    sigma = np.sqrt(np.log(1 + return_std**2 / (1 + return_mean)**2))
    
    for sim in range(simulations):
        portfolio = initial_portfolio
        for year in range(years):
            current_withdrawal = initial_withdrawal * (1 + inflation_rate)**year
            max_withdrawal = min(initial_withdrawal * 3 * (1 + inflation_rate)**year, 1200000 * (1 + inflation_rate)**year)
            
            # 動態提款策略
            if portfolio > initial_portfolio * (1 + inflation_rate)**year:
                withdrawal = min(max(current_withdrawal, portfolio * withdrawal_rate), max_withdrawal)
            else:
                withdrawal = min(current_withdrawal, portfolio * 0.03)  # 減少提款率
            
            all_withdrawals[sim, year] = withdrawal
            portfolio -= withdrawal
            portfolio += additional_income
            
            # 使用對數正態分佈生成年度回報率，並限制範圍
            annual_return = np.clip(np.random.lognormal(mu, sigma) - 1, -0.4, 0.4)
            portfolio *= (1 + annual_return)
            
            all_portfolios[sim, year+1] = portfolio
            
            if portfolio <= emergency_fund:
                results.append(False)
                final_portfolios.append(portfolio)
                all_portfolios[sim, year+1:] = portfolio
                break
            
            if year == years - 1:
                results.append(True)
                final_portfolios.append(portfolio)
    
    success_count = sum(results)
    success_rate = success_count / len(results)
    median_final_portfolio = np.median(final_portfolios)
    
    min_withdrawals = np.min(all_withdrawals, axis=0)
    max_withdrawals = np.max(all_withdrawals, axis=0)
    
    return success_rate, success_count, simulations - success_count, final_portfolios, median_final_portfolio, all_portfolios, min_withdrawals, max_withdrawals

# 定義模擬參數
initial_portfolio = 20000000
years = 50
withdrawal_rate = 0.04
inflation_rate = 0.02
return_mean = 0.07
return_std = 0.15
simulations = 10000
initial_withdrawal = 400000
additional_income = 200000
emergency_fund = 1000000

# 執行模擬
success_rate, successes, failures, final_portfolios, median_final_portfolio, all_portfolios, min_withdrawals, max_withdrawals = monte_carlo_retirement_simulation(
    initial_portfolio=initial_portfolio,
    years=years,
    withdrawal_rate=withdrawal_rate,
    inflation_rate=inflation_rate,
    return_mean=return_mean,
    return_std=return_std,
    simulations=simulations,
    initial_withdrawal=initial_withdrawal,
    additional_income=additional_income,
    emergency_fund=emergency_fund
)

# 計算成長率
growth_rates = (all_portfolios[:, -1] / all_portfolios[:, 0]) ** (1/years) - 1

# 打印結果和統計信息
print(f"退休成功率: {success_rate:.2%}")
print(f"成功次數: {successes}")
print(f"失敗次數: {failures}")
print(f"資產中位數: {median_final_portfolio:,.0f} 台幣")

final_percentiles = np.percentile(all_portfolios[:, -1], [10, 25, 50, 75, 90])
growth_rates_percentiles = np.percentile(growth_rates, [0, 25, 50, 75, 100])
max_final_asset = np.max(all_portfolios[:, -1])
min_final_asset = np.min(all_portfolios[:, -1])
positive_final_assets = np.sum(all_portfolios[:, -1] > 0)

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
print(f"50年後仍有正資產的模擬次數: {positive_final_assets}")

print(f"\n初始提款: {initial_withdrawal:,.0f} 台幣")
print(f"50年後預估最小提款: {min_withdrawals[-1]:,.0f} 台幣")
print(f"50年後預估最大提款: {max_withdrawals[-1]:,.0f} 台幣")

# 視覺化結果
plt.figure(figsize=(20, 15))

# 視覺化結果（圖表代碼保持不變）
# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 視覺化結果
plt.figure(figsize=(20, 15))

# 1. 退休計劃成功率柱狀圖
plt.subplot(3, 3, 1)
plt.bar(['成功', '失敗'], [success_rate, 1-success_rate], color=['green', 'red'])
plt.title('退休計劃成功率')
plt.ylabel('概率')
plt.ylim(0, 1)

# 2. 最終資產分佈直方圖
plt.subplot(3, 3, 2)
plt.hist(final_portfolios, bins=50, edgecolor='black')
plt.title('最終資產分佈')
plt.xlabel('資產價值 (台幣)')
plt.ylabel('頻率')

# 3. 最終資產箱形圖
plt.subplot(3, 3, 3)
plt.boxplot(final_portfolios)
plt.title('最終資產箱形圖')
plt.ylabel('資產價值 (台幣)')

# 4. 最終資產累積分佈函數
plt.subplot(3, 3, 4)
plt.hist(final_portfolios, bins=50, cumulative=True, density=True, histtype='step')
plt.title('最終資產累積分佈函數')
plt.xlabel('資產價值 (台幣)')
plt.ylabel('累積概率')

# 5. 年度資產變化
percentiles = [10, 25, 50, 75, 90]
plt.subplot(3, 3, 5)
for p in percentiles:
    plt.plot(np.percentile(all_portfolios, p, axis=0), label=f'{p}th percentile')
plt.title('年度資產變化')
plt.xlabel('年')
plt.ylabel('資產價值 (台幣)')
plt.legend()

# 6. 資產成長率分佈
growth_rates = (all_portfolios[:, -1] / all_portfolios[:, 0]) ** (1/years) - 1
plt.subplot(3, 3, 6)
plt.hist(growth_rates, bins=50, edgecolor='black')
plt.title('年化資產成長率分佈')
plt.xlabel('年化成長率')
plt.ylabel('頻率')

# 7. 風險值 (Value at Risk) 分析
var_95 = np.percentile(final_portfolios, 5)
var_99 = np.percentile(final_portfolios, 1)
plt.subplot(3, 3, 7)
plt.hist(final_portfolios, bins=50, edgecolor='black')
plt.axvline(var_95, color='r', linestyle='dashed', label='95% VaR')
plt.axvline(var_99, color='g', linestyle='dashed', label='99% VaR')
plt.title('風險值 (VaR) 分析')
plt.xlabel('最終資產價值 (台幣)')
plt.ylabel('頻率')
plt.legend()

# 8. 資產軌跡
plt.subplot(3, 3, 8)
for i in range(min(100, simulations)):  # 繪製前100條軌跡
    plt.plot(all_portfolios[i, :], alpha=0.1)
plt.title('資產軌跡 (100次模擬)')
plt.xlabel('年')
plt.ylabel('資產價值 (台幣)')
# 9. 提款金額變化
years_range = range(years)
plt.subplot(3, 3, 9)
plt.plot(years_range, min_withdrawals, label='最小提款')
plt.plot(years_range, max_withdrawals, label='最大提款')
plt.title('年度提款金額範圍')
plt.xlabel('年')
plt.ylabel('提款金額 (台幣)')
plt.legend()

plt.tight_layout()
plt.show()

# 額外統計信息
var_95 = np.percentile(final_portfolios, 5)
var_99 = np.percentile(final_portfolios, 1)
print(f"\n年化資產成長率中位數: {np.median(growth_rates):.2%}")
print(f"95% 風險值 (VaR): {var_95:,.0f} 台幣")
print(f"99% 風險值 (VaR): {var_99:,.0f} 台幣")