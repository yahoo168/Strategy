import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import sqrt
from .utils import *

# 計算平均年化手續費
def cal_avg_commision(fee_series, freq):
    freq_days = freq_to_days(freq)
    return fee_series.mean() * (252/freq_days)

# 計算基本績效指標（平均報酬 / 平均波動 / 平均夏普）
def cal_basic_performance(sum_percent_series, period=252):
    #計算rolling平均報酬
    return_rolling = (1+sum_percent_series).rolling(period).apply(np.prod, raw=True) - 1
    #計算rolling平均波動率
    volatility_rolling = sum_percent_series.rolling(period).std() * sqrt(252)
    #計算平均夏普ratio
    sharpe_rolling = return_rolling / volatility_rolling
    avg_return = return_rolling.mean()
    avg_volatility = volatility_rolling.mean()
    avg_sharpe = sharpe_rolling.mean()

    return avg_return, avg_volatility, avg_sharpe

# 計算MDD，回傳MDD與 起始日 / 結束日
def cal_maxdrawdown(value_series):
    #紀錄近期資產價值最高的日期，預設為第一日
    peak_date = value_series.index[0]
    #紀錄近期資產價值最高的數額，預設為1
    peak_value = 1
    #紀錄MDD期間的起始日與結束日
    MDD_s_date = value_series.index[0]
    MDD_e_date = value_series.index[0]
    MDD = 0

    for date, v in value_series.items():
        #若價值突破前波高點，則更新高點日期與高點價值
        if v >= peak_value:
            peak_date = date
            peak_value = v
        
        #反之則紀錄期間內最大跌幅
        else:
            drawdown = (peak_value - v) / peak_value
            if drawdown >= MDD:
                MDD_s_date = peak_date
                MDD_e_date = date
                MDD = drawdown

    # 回傳MDD起始日（起跌日）與MDD結束日（谷底）
    return(round(MDD,2), (MDD_s_date, MDD_e_date))

# 待改：有更直觀的作法，參考如下：
# def MaxDrawdown(return_series: pd.Series):
#     """Takes a time series of asset returns.
#        Return:
#            1. MDD
#            2. MDD's Date
#     """
#     wealth_index = 1000*(1+return_series).cumprod()
#     previous_peaks = wealth_index.cummax()
#     drawdowns = (wealth_index - previous_peaks)/previous_peaks
#     return drawdowns.min(), drawdowns.idxmin()

#計算勝率，每日比對策略與大盤的百分比變動高低
def cal_win_rate(sum_percent_series, benchmark_percent_series):
    # 因sum_percent_series在拆分區間進行處理，會導致合成後有精度誤差
    # 若單一標的以自身為benchmark，會出現win_rate>0的狀況，故設置ERROR_THERESHOLD避免
    ERROR_THERESHOLD = 0.000000000000001
    win_series = sum_percent_series > (benchmark_percent_series + ERROR_THERESHOLD)
    win_rate = round((win_series.sum() / len(win_series)), 2)
    cum_win_series = (sum_percent_series - benchmark_percent_series)
    
    return win_rate

#計算夏普勝率，每日比對策略與大盤的夏普值高低
def cal_sharpe_win_rate(sum_percent_series, benchmark_percent_series, period=252):
    #計算rolling平均報酬
    sum_return_rolling = (1+sum_percent_series).rolling(period).apply(np.prod, raw=True) - 1
    #計算rolling平均波動率
    sum_volatility_rolling = sum_percent_series.rolling(period).std() * sqrt(252)
    #計算平均夏普ratio
    sum_sharpe_rolling = sum_return_rolling / sum_volatility_rolling

    #計算rolling平均報酬
    benchmark_return_rolling = (1+benchmark_percent_series).rolling(period).apply(np.prod, raw=True) - 1
    #計算rolling平均波動率
    benchmark_volatility_rolling = benchmark_percent_series.rolling(period).std() * sqrt(252)
    #計算平均夏普ratio
    benchmark_sharpe_rolling = benchmark_return_rolling / benchmark_volatility_rolling

    sharpe_win_rolling = sum_sharpe_rolling > benchmark_sharpe_rolling 
    sharpe_win_days = sharpe_win_rolling.sum()
    sharpe_win_rate = sharpe_win_days / len(sharpe_win_rolling)
    
    return round(sharpe_win_rate, 2)

#計算分年績效
def cal_performance_per_yr(sum_percent_series, benchmark_percent_series):
    #先取得資產組合的分年序列
    dt = sum_percent_series
    sum_percent_series_list = [dt[dt.index.year == y] for y in dt.index.year.unique()]    
    
    #取得Benchmark的分年序列
    dt_2 = benchmark_percent_series
    benchmark_percent_series_list = [dt_2[dt_2.index.year == y] for y in dt_2.index.year.unique()]
    
    #紀錄分年績效
    perfomance_per_yr_df = pd.DataFrame()
    for index, percent_series in enumerate(sum_percent_series_list,0):
        #以各年序列第一筆資料的年份作為本序列年份代表
        yr = percent_series.index[0].year
        
        #使用原先計算全區間的績效指標函數，將period設為分段資料長度，
        #則只有最後一日的rolling績效指標有值，其他日為Nan
        yr_return, yr_std, yr_sharpe = cal_basic_performance(percent_series, period=len(percent_series))
        
        #由百分比序列構建價值序列，以便計算績效
        value_series = (1+percent_series).cumprod()
        MDD, date_interval = cal_maxdrawdown(value_series)
        MDD_s_date =  datetime.strftime(date_interval[0], "%Y-%m-%d")
        MDD_e_date =  datetime.strftime(date_interval[1], "%Y-%m-%d")
        win_rate = cal_win_rate(percent_series, benchmark_percent_series_list[index])
        
        perfomance_per_yr_df.loc[yr, "Return"] = round(yr_return, 2)
        perfomance_per_yr_df.loc[yr, "Std"] = round(yr_std, 2)
        perfomance_per_yr_df.loc[yr, "Sharpe"] = round(yr_sharpe, 2)
        perfomance_per_yr_df.loc[yr, "MDD"] = round(MDD, 2)
        perfomance_per_yr_df.loc[yr, "MDD_s_date"] = MDD_s_date
        perfomance_per_yr_df.loc[yr, "MDD_e_date"] = MDD_e_date
        perfomance_per_yr_df.loc[yr, "Win Rate"] = win_rate
    
    return perfomance_per_yr_df

#將單年的百分比序列切分為分季的百分比序列（依月份區分）
#傳入值為一個list，其中含有分年的百分比變動序列。
def _yr_to_quarter_series(yr_series_list):
    quarter_series_list = list()
    for series_yr in yr_series_list:
        Qtr_1_series = series_yr[(series_yr.index.month > 0)&(series_yr.index.month <= 3)]
        Qtr_2_series = series_yr[(series_yr.index.month > 3)&(series_yr.index.month <= 6)]
        Qtr_3_series = series_yr[(series_yr.index.month > 6)&(series_yr.index.month <= 9)]
        Qtr_4_series = series_yr[(series_yr.index.month > 9)&(series_yr.index.month <= 12)]
        
        quarter_series_list.append(Qtr_1_series)
        quarter_series_list.append(Qtr_2_series)
        quarter_series_list.append(Qtr_3_series)
        quarter_series_list.append(Qtr_4_series)
    
    return quarter_series_list

#計算分季績效
def cal_performance_per_quarter(sum_percent_series, benchmark_percent_series):
    dt = sum_percent_series
    #先取得資產組合的分年序列
    sum_percent_yr_series_list = [dt[dt.index.year == y] for y in dt.index.year.unique()]
    #取得分季序列
    sum_percent_quarter_series_list = _yr_to_quarter_series(sum_percent_yr_series_list)
    #去除分季序列的空值，因頭尾兩年可能沒有完整的分季資料
    sum_percent_quarter_series_list = [x for x in sum_percent_quarter_series_list if not x.empty]
    
    #方法同上，取得Benchmark的分季序列
    dt_2 = benchmark_percent_series
    benchmark_percent_yr_series_list = [dt_2[dt_2.index.year == y] for y in dt_2.index.year.unique()]
    benchmark_percent_quarter_series_list = _yr_to_quarter_series(benchmark_percent_yr_series_list)
    benchmark_percent_quarter_series_list = [x for x in benchmark_percent_quarter_series_list if not x.empty]

    #紀錄分季績效的DataFrame
    perfomance_per_Qr_df = pd.DataFrame()
    for index, percent_series in enumerate(sum_percent_quarter_series_list, 0):
        #以分季序列第一日的季度代號作為該季的代表（e.g:2021Q1）
        Qr = pd.Period(percent_series.index[0], 'Q').__str__()
        
        #計算分季績效的方式與分年績效相同
        Qr_return, Qr_std, Qr_sharpe = cal_basic_performance(percent_series, period=len(percent_series))
        value_series = (1+percent_series).cumprod()
        MDD, date_interval = cal_maxdrawdown(value_series)
        MDD_s_date = datetime.strftime(date_interval[0], "%Y-%m-%d")
        MDD_e_date = datetime.strftime(date_interval[1], "%Y-%m-%d")
        win_rate = cal_win_rate(percent_series, benchmark_percent_quarter_series_list[index])
        
        #填入分季績效指標
        perfomance_per_Qr_df.loc[Qr, "Return"] = round(Qr_return, 2)
        perfomance_per_Qr_df.loc[Qr, "Std"] = round(Qr_std, 2)
        perfomance_per_Qr_df.loc[Qr, "Sharpe"] = round(Qr_sharpe, 2)
        perfomance_per_Qr_df.loc[Qr, "MDD"] = round(MDD, 2)
        perfomance_per_Qr_df.loc[Qr, "MDD_s_date"] = MDD_s_date
        perfomance_per_Qr_df.loc[Qr, "MDD_e_date"] = MDD_e_date
        perfomance_per_Qr_df.loc[Qr, "Win Rate"] = win_rate
    
    return perfomance_per_Qr_df

#計算分月績效
def cal_performance_per_month(sum_percent_series, benchmark_percent_series):
    #先取得資產組合的分年序列
    sum_percent_month_series_list = list()
    dt = sum_percent_series
    sum_percent_yr_series_list = [dt[dt.index.year == y] for y in dt.index.year.unique()]    
    #由分年序列取得每年的分月序列
    for yr_series in sum_percent_yr_series_list:
        for y in yr_series.index.month.unique():
            sum_percent_month_series_list.append(yr_series[yr_series.index.month == y])
        

    #取得Benchmark的分年序列
    benchmark_percent_month_series_list = list()
    dt_2 = benchmark_percent_series
    #由分年序列取得每年的分月序列
    benchmark_percent_series_list = [dt_2[dt_2.index.year == y] for y in dt_2.index.year.unique()]
    for yr_series in benchmark_percent_series_list:
        for y in yr_series.index.month.unique():
            benchmark_percent_month_series_list.append(yr_series[yr_series.index.month == y])
    
    #紀錄分月績效
    perfomance_per_month_df = pd.DataFrame()
    for index, percent_series in enumerate(sum_percent_month_series_list,0):
        #以各年序列第一筆資料的“年份_月份"作為本序列時間戳記代表
        month = str(percent_series.index[0].year)+'_'+str(percent_series.index[0].month)
        
        #使用原先計算全區間的績效指標函數，將period設為分段資料長度，
        #則只有最後一日的rolling績效指標有值，其他日為Nan
        #待改：似乎不太直觀
        month_return, month_std, month_sharpe = cal_basic_performance(percent_series, period=len(percent_series))
        
        #由百分比序列構建價值序列，以便計算績效
        value_series = (1+percent_series).cumprod()
        MDD, date_interval = cal_maxdrawdown(value_series)
        MDD_s_date =  datetime.strftime(date_interval[0], "%Y-%m-%d")
        MDD_e_date =  datetime.strftime(date_interval[1], "%Y-%m-%d")
        win_rate = cal_win_rate(percent_series, benchmark_percent_month_series_list[index])
        
        perfomance_per_month_df.loc[month, "Return"] = month_return
        perfomance_per_month_df.loc[month, "Std"] = month_std
        perfomance_per_month_df.loc[month, "Sharpe"] = month_sharpe
        perfomance_per_month_df.loc[month, "MDD"] = MDD
        perfomance_per_month_df.loc[month, "MDD_s_date"] = MDD_s_date
        perfomance_per_month_df.loc[month, "MDD_e_date"] = MDD_e_date
        perfomance_per_month_df.loc[month, "Win Rate"] = win_rate
    
    return perfomance_per_month_df

#計算調倉區間內，個別資產對整體組合變動的貢獻佔比（有可能超過100%）
def cal_profit_ratio(percent_df=None, freq="month"):
    period = freq_to_days(freq)
    sum_percent_series = percent_df.sum(axis=1) 
    rolling_percent_df = percent_df.rolling(period).mean()
    ratio_df = rolling_percent_df.div(sum_percent_series, axis=0)
    ratio_df.dropna(axis=0, inplace=True)
    
    #部分極端值會嚴重影響圖表呈現，限制極端值範圍後再展示
    de_extreme_ratio_df = ratio_df.copy()
    de_extreme_ratio_df[de_extreme_ratio_df > 1] = 1
    de_extreme_ratio_df[de_extreme_ratio_df < -1] = -1
    
    return de_extreme_ratio_df, ratio_df

# 分析持倉各指標
def analyze_weight_df(self, weight_df, rank_nums):
    weight_analysis_dict = dict()
    # 各標的平均持倉
    avg_weight_per_ticker_df = weight_df.mean().sort_values(ascending=False).apply(lambda x: str(round(100*x,2))+"%")
    # 只顯示平均前N大持倉
    avg_weight_per_ticker_df = avg_weight_per_ticker_df.to_frame(name="Weight").head(rank_nums)
    
    # 持倉檔數
    max_holding_nums = (weight_df>0).sum(axis=1).max()
    min_holding_nums = (weight_df>0).sum(axis=1).min()
    mean_holding_nums = (weight_df>0).sum(axis=1).mean()
    
    # 計算前N大持倉平均權重
    weight_rank_df = weight_df.rank(axis=1, ascending=False)
    filtered_weight_list = list()
    for rank_num in range(1, rank_nums+1):
        filtered_df = weight_df[weight_rank_df == rank_num]
        filtered_weight = filtered_df.sum(axis=1).mean()
        filtered_weight_list.append(round(filtered_weight, 2))

    # 前N大持倉平均權重
    N_rank_weight_series = pd.Series(filtered_weight_list, index=range(1, rank_nums+1))
    # 換倉率序列 
    turnover_rate_series = abs(weight_df.diff()).sum(axis=1)
    # 換倉區間的平均日數
    avg_changeDate_interval_days = pd.Series(weight_df.index).diff().mean().days
    # 年化換手率(待改：是否應以期末價格影響後的權重計算？)
    annualized_turnover_rate = turnover_rate_series.mean() * (252/avg_changeDate_interval_days)
    # 總倉位序列 
    weight_series = weight_df.sum(axis=1)

    weight_analysis_dict["avg_weight_per_ticker"] = avg_weight_per_ticker_df
    weight_analysis_dict["max_holding_nums"] = max_holding_nums
    weight_analysis_dict["min_holding_nums"] = min_holding_nums
    weight_analysis_dict["mean_holding_nums"] = mean_holding_nums
    weight_analysis_dict["N_rank_weight_series"] = N_rank_weight_series
    weight_analysis_dict["turnover_rate_series"] = turnover_rate_series
    weight_analysis_dict["annualized_turnover_rate"] = annualized_turnover_rate
    weight_analysis_dict["weight_series"] = weight_series
    return weight_analysis_dict