import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
import pandas as pd
import multiprocessing as mp
from functools import partial
class ResultAnalysis:

    @staticmethod
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

    @staticmethod
    def plot_results(initial_portfolio,success_rate, final_portfolios, all_portfolios, min_withdrawals, max_withdrawals, 
                     avg_withdrawals, zero_withdrawal_percentage, years, simulations):
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