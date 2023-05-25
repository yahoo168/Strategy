from .strategy import *

class Factor(Strategy):    
    def trans_rank_to_weight(self, rank_df, rank_low_bar, rank_high_bar, weight_method="Equal", total_weight=1):        
        # 等權重
        if weight_method == "Equal":
            signal_df = (rank_df >= rank_low_bar) & (rank_df < rank_high_bar).astype(int)
        
        # 權重隨Rank值下降呈指數下降，ex：Rank1：100%、Rank2：50%、Rank3：25%
        elif weight_method == "Expo":
            signal_df = rank_df[(rank_df >= rank_low_bar) & (rank_df < rank_high_bar)]
            signal_df = signal_df.fillna(0)
            signal_df = 1/signal_df
        
        # 權重隨Rank值上升呈指數下降，ex：Rank3：100%、Rank2：50%、Rank1：25%
        elif weight_method == "Anti-Expo":
            signal_df = rank_df[(rank_df >= rank_low_bar) & (rank_df < rank_high_bar)]
            signal_df = signal_df.fillna(0)

        # 將異常值填補為0；待改：未必要如此
        signal_df = signal_df.replace([np.inf, -np.inf], 0)
        adjust_factor = signal_df.sum(axis=1)
        # 先將權重線性縮放至總和為1
        weight_df = signal_df.div(adjust_factor, axis=0)
        # 再依照total_weight調整（若預設為1即沒有變化）
        weight_df = weight_df * total_weight
        # 填補權重缺值為0（待改：應該輸出異常檢測）
        weight_df = weight_df.fillna(0)
        # 取得換倉時點，並調整換倉頻率
        change_tradeDate_list = self._get_change_tradeDate_list(rebalance_freq=self.options["freq"])
        return weight_df.loc[change_tradeDate_list,:]

    def get_other_factor_df(self, Factor_Class):
        Factor = Factor_Class(self.universe_ticker_list, self.options)
        return Factor.cal_factor_df()
    
    # 預設最高者rank為1
    def get_cs_rank(self, df):
        return df.rank(axis=1, ascending=False)
    
    def cal_factor_performance(self, ascending=False, group_nums=10 , show=True, weight_method="Equal", benchmark="Equal"):
        factor_df = self.cal_factor_df()
        # 預設factor值最高者rank為1
        rank_df = self.get_cs_rank(factor_df)
        benchmark_percent_series = self._get_benchmark_values(rank_df, benchmark)
        
        backtest_per_group_list = list()
        for group in range(group_nums):
            if show:
                logging.info("第{n}分組計算中".format(n=group))
            
            per_group_ticker_nums = len(rank_df.columns) / group_nums
            rank_low_bar = round(group * per_group_ticker_nums)
            rank_high_bar = round((group+1) * per_group_ticker_nums)
            # 將因子rank轉化爲權重矩陣，預設為等權重，且線性縮放至滿倉位為1
            weight_df = self.trans_rank_to_weight(rank_df, rank_low_bar, rank_high_bar, weight_method=weight_method)
            time_mask = (weight_df.index >= self.options["start_date"]) & \
                        (weight_df.index <= self.options["end_date"])
            
            weight_df = weight_df[time_mask]
            # 權重檢測是否異常（檢測過程詳見backtest.py)
            self._check_weight_df(weight_df, self.options)
            backtest = self._cal_pnl(weight_df)
            backtest_per_group_list.append(backtest)
        
        performance_df, extra_percent_series_list = self.evaluate_factor(backtest_per_group_list, benchmark_percent_series)
        print(performance_df)
        return backtest_per_group_list, performance_df, benchmark_percent_series
    
    def _get_benchmark_values(self, rank_df, benchmark):
        weight_df = rank_df.copy()
        if benchmark == "Equal":
            weight_df.values[:] = 1
            weight_df = adjust_weight(weight_df, total_weight=1)

        elif benchmark == "None":
            weight_df.values[:] = 0
            weight_df = adjust_weight(weight_df, total_weight=0)
        
        backtest = self._cal_pnl(weight_df)
        return backtest.sum_percent_series

    def _cal_pnl(self, weight_df):
        #檢查策略輸出的權重矩陣是否符合基本規範
        self._check_weight_df(weight_df, self.options)
        #使用權重矩陣建立回測物件
        backtest = Backtest(weight_df, self.db, self.options)
        #計算回測損益（關閉顯示回測計算過程）
        backtest.activate(show=False)
        #評估績效
        backtest.evaluate(show=False)
        return backtest

    def evaluate_factor(self, backtest_per_group_list, benchmark_percent_series):
        sum_value_per_group_df = pd.DataFrame()
        group_nums = len(backtest_per_group_list)
        performance_indicators_per_group = pd.DataFrame(index=list(range(group_nums)))
        extra_percent_series_list = list()
        for group in range(group_nums):
            backtest = backtest_per_group_list[group]
            
            performance_dict = backtest.performance
            avg_return = performance_dict["Average Annualized Return"]            
            avg_volatility = performance_dict["Average Annualized Volatility"]
            avg_sharpe = performance_dict["Average Sharpe"]
            MDD = performance_dict["MDD"]
            date_interval = performance_dict["MDD INRERVAL"]
            # performance_per_Yr_df = performance_dict["Performance Per Yr"]
            # performance_per_Qr_df = performance_dict["Performance Per Qr"]
        
            # 填入全區間，因子分組績效指標
            performance_indicators_per_group.loc[group, "Return"] = round(avg_return, 2)
            performance_indicators_per_group.loc[group, "Std"] = round(avg_volatility, 2)
            performance_indicators_per_group.loc[group, "Sharpe"] = round(avg_sharpe, 2)
            performance_indicators_per_group.loc[group, "MDD"] = round(MDD, 2)
            #performance_indicators_per_group.loc[group, "MDD_s_date"] = date_interval[0]
            #performance_indicators_per_group.loc[group, "MDD_e_date"] = date_interval[1]
        
            extra_percent_series = (backtest.sum_percent_series - benchmark_percent_series)
            sum_value_per_group_df[group] = (extra_percent_series+1).cumprod()
            
            periods = 252
            annualized_extra_return_mean = ((1+extra_percent_series).rolling(periods, min_periods=1).apply(np.prod, raw=True) - 1).mean() * (252/periods)
            annualized_extra_return_vol = (extra_percent_series.rolling(periods, min_periods=1).std() * sqrt(252)).mean()
            annualized_IR =  annualized_extra_return_mean / annualized_extra_return_vol
            
            performance_indicators_per_group.loc[group, "Extra_Return"] = round(annualized_extra_return_mean, 2)
            performance_indicators_per_group.loc[group, "Extra_Vol"] = round(annualized_extra_return_vol, 2)
            performance_indicators_per_group.loc[group, "IR"] = round(annualized_IR, 2)

            extra_percent_series_list.append(extra_percent_series)
            # print(annualized_extra_return_mean)
            # print(annualized_extra_return_vol)
            # print(annualized_extra_return_mean / annualized_extra_return_vol)

        #factor_percent_series = (extra_percent_series_list[3] - extra_percent_series_list[0])
        #factor_percent_series = (extra_percent_series_list[3])
        #print(performance_indicators_per_group)
        #print((factor_percent_series+1).cumprod())
        #(factor_percent_series+1).cumprod().plot()
        #plt.show()
        return performance_indicators_per_group, extra_percent_series_list