from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json, pickle
import time
from self_finance_database_tool import *
from .evaluation import *
from .utils import *

class Backtest(object):
    """(待改：)
    Backtest為一回測物件，必要的輸入值為策略類別與設定回測選項（options）
    
    物件方法概覽：
        & activate: 呼叫策略以取得權重df，並計算此權重在過去的每日價值變動
        & evaluate: 依照計算完成的每日百分比變動計算績效指標
        & save_record: 將各績效指標以及策略參數存檔
    """
    def __init__(self, weight_df:pd.DataFrame, options:dict):
        self.options = options
        
        #透過路徑建立資料庫物件，以便取用資料
        self.db = Database(self.options["database_path"])
        self.weight_df = weight_df
        
        #調整原始設定的日期，改為最接近該日的下一個交易日（若該日即為交易日則不變）
        # self.start_date = self.db.get_next_tradeDate(datetime.strptime(self.options["start_date"], "%Y-%m-%d"), country=self.options["country"])
        # self.end_date = self.db.get_next_tradeDate(datetime.strptime(self.options["end_date"], "%Y-%m-%d"), country=self.options["country"])
        self.start_date = self.options["start_date"]
        self.end_date = self.options["end_date"]

        #存放Benchmark的各類資料，便於績效比對。
        self.benchmark_data_dict = dict()

        #存放投資組合績效評估的各類資料，便於績效比對。
        self.performance = dict()
   
        #紀錄每次換倉的手續費序列
        self.performance["commission_fee_series"] = pd.Series()

    #若需要將原策略decay測試，則調節權重矩陣的Index（交易日）
    def _decay_weight_df(self):
        decay_days = self.options["decay_days"]
        if decay_days != 0:
            #正常情況下decay_days不應為負數，但在研究階段可能須測試forward looking的影響程度
            logging.info("注意:目前Decay {decay_days}天".format(decay_days=decay_days))
            #先加/減原始交易日後，調整到最接近該日的下一個交易日，覆蓋原先權重矩陣的index Series
            next_date_series = (self.weight_df).index + timedelta(days=decay_days)
            self.weight_df.index = pd.Index(map(lambda x:self.db.get_next_tradeDate(x, country=self.options["country"]), next_date_series))

    #以調倉日拆分數個持倉區間，以計算報酬
    def _cal_return(self):
        #回測最後一日若沒有指定權重，則以權重矩陣的最後一次指定權重作為最終權重(會多扣一次手續費，但計算績效較方便)
        if self.end_date not in self.weight_df.index:
            logging.info("回測期間最後一日({date1})的權重未指定，故以策略最後指定權重之日({date2})為準以自動補足"
                    .format(date1=self.end_date.strftime("%Y-%m-%d"), date2=self.weight_df.index[-1].strftime("%Y-%m-%d")))

            #Replicate the last row
            self.weight_df = self.weight_df.append(self.weight_df.iloc[-1,:])
            date_index_list = self.weight_df.index.tolist()
            #change the index of the last row
            date_index_list[-1] = self.end_date
            self.weight_df.index = pd.Index(date_index_list)
        
        changeDate_list = self.weight_df.index
        holder_ticker_list = self.weight_df.columns
        
        #從策略權重矩陣取得持有標的列表，以此取得調整價
        self.adjclose_df = self.db.get_universe_df(holder_ticker_list, 
             data_type="adjclose", start_date=self.start_date, end_date=self.end_date, data_format="all")
        
        #因應部分策略每日皆有給定權重，可不進行矩陣分拆，以加速計算
        #待改：上面的權重矩陣調整會影響此部分(因在回測期末多加一日權重，導致長度不一)
        
        pct_change_df = self.adjclose_df.pct_change()
        if len(pct_change_df) == len(self.weight_df):
            self.sum_percent_series = (pct_change_df * self.weight_df).sum(axis=1)
            weight_diff_df = self.weight_df.shift(1) - self.weight_df
            self.performance["Commission Fee Series"] = (abs(weight_diff_df) * self.options["commission_rate"]).fillna(0).values
            #透過百分比變動計算資產組合的日報酬，資產初始金額由options給定
            self.sum_value_series = (self.sum_percent_series+1).cumprod() * self.options["initial_value"]    
        
        #非每日調倉策略
        else:
            #以兩調倉日作為區間起始，製作時間區間列表，透過self._cal_interval_return計算各區間績效後再全部合併       
            cal_interval = [(changeDate_list[i-1], changeDate_list[i]) for i in range(1, len(changeDate_list))]
            
            #紀錄各區間每日總資產變動的百分比
            interval_sum_percent_series_list = list()

            #紀錄各區間最後轉倉的手續費（佔總資產價值比例）#待改：最後一個區間的手續費？？
            interval_commission_fee_list = list()
            
            #待改：之後用multi-thread做
            
            for index, interval in enumerate(cal_interval, 1):
                percentage = 100*index/len(cal_interval)
                sys.stdout.write('\r'+"回測計算：完成度{percentage:.2f}%".format(percentage=percentage))
                #在計算percent_series已在最後一天扣除手續費，只是另外獨立紀錄
                interval_sum_percent_series, interval_commission_fee = self._cal_interval_return(interval)
                interval_sum_percent_series_list.append(interval_sum_percent_series)
                interval_commission_fee_list.append(interval_commission_fee)
            print()
            
            #待改：為啥不直接放進self.performance
            #合成各換倉區間回測所得的百分比變動序列
            self.sum_percent_series = pd.concat(interval_sum_percent_series_list)
            self.performance["Commission Fee Series"] = pd.Series(interval_commission_fee_list, index=changeDate_list[1:])

            #透過百分比變動計算資產組合的日報酬，資產初始金額由options給定
            self.sum_value_series = (self.sum_percent_series+1).cumprod() * self.options["initial_value"]    
  
    # 距離期初日期與結束日期，各自最近的下一個交易日，也應在權重矩陣的調倉日中
    # 計算換倉區間的績效變動，換倉日前一日收盤後換倉，隔日為新倉位。
    # 回傳該區間的每日資產價值百分比變動（pd.Series）以及期末換倉日的手續費(float)
    def _cal_interval_return(self, interval):
        #從interval讀取此區間期初與結束的原始日期
        s_date, e_date = interval[0], interval[1]
        
        #將期初日期與結束日期各往前調一個交易日，以實現換倉日當天開盤前即換完倉
        #意即回測區間的第一日，便應計算該日的漲跌幅，而最後一日不計算（由下一個區間段處理）
        s_date_last_tradeDate = self.db.get_last_tradeDate(s_date-timedelta(days=1), country=self.options["country"])
        e_date_last_tradeDate = self.db.get_last_tradeDate(e_date-timedelta(days=1), country=self.options["country"])
        
        #取得期初日（回測開始日 或初始換倉日）的目標權重
        s_weight = self.weight_df.loc[s_date,:]
        
        #取得結束日（下一次換倉日）的目標權重以便計算調倉手續費
        next_weight = self.weight_df.loc[e_date,:]
        
        #取得現金比例以便計算調倉手續費
        cash_weight = max(1 - abs(s_weight).sum(), 0)
        
        #截取該區間的價格df（往起始日前多取一天，以取得起始日當日的百分比變動）
        mask = (self.adjclose_df.index >= s_date_last_tradeDate) & (self.adjclose_df.index <= e_date_last_tradeDate)
        percent_df = self.adjclose_df.pct_change()[mask]
        
        #將起始日的前一日變動設為0，以避免起始日的百分比變動出錯
        #待改：看不懂，直接刪掉不就好了？，但我記得這樣跑起來才是對的
        percent_df.fillna(0, inplace=True)
        
        holder_ticker_list = percent_df.columns
        
        #待改：如何正確處理空部位？
        #將各部位績效取連乘，並乘上起始日的權重分配，得出本次換倉區間的加權累積報酬
        long_short_direction = (s_weight > 0).astype(int)
        long_short_direction[long_short_direction == 0] = -1
        weighted_cumprod_df = ((percent_df*long_short_direction)+1).cumprod() * abs(s_weight)
    
        #須加計現金比重，因現金部位的價值不變
        sum_cumprod_series = weighted_cumprod_df.sum(axis=1) + cash_weight
        sum_percent_series = sum_cumprod_series.pct_change()
        #print(s_date_last_tradeDate)
        #print(percent_df)
        #去除換倉日前一交易日的資料（原是為配合計算百分比變動而加入）
        sum_percent_series = sum_percent_series[1:]
        
        # 待改：考量放空的情況
        # neg_comprod_df = (-percent_df[neg_weight.index]+1).cumprod()
        
        commission_fee = 0
        #有可能權重區間段內皆為假期，不存在任何報價會導致sum_percent_series為空
        if len(sum_percent_series) > 0:
            commission_fee = self._cal_commision_fee(s_weight, cash_weight, weighted_cumprod_df.iloc[-1,:], next_weight) 
            #計算出換倉手續費後在最後一日扣除
            sum_percent_series[-1] -= commission_fee
        
        return sum_percent_series, commission_fee

    #計算換倉交易成本（手續費)
    def _cal_commision_fee(self, s_weight, cash_weight, e_value, next_weight):
        #計算因本換倉期間因價格變動所導致的權重偏移
        commission_fee = 0
        if e_value.sum() != 0:
            #以待調整權重佔總部位比例計算手續費
            #weighted_cumprod_df.iloc[-1,:]為加權後期末的各標的價值，可藉此算出實際的期末權重
            e_weight = e_value / (e_value.sum() + cash_weight)
            
            #期末權重 - 目標權重 = 須變動的比例，以此比例計算手續費佔總資產組合價值的比例
            #因原始手續費（0.004)為買進賣出的總數，單方向交易僅一半
            commission_series = abs(e_weight - next_weight) * (self.options["commission_rate"])/2

            #commission_series為各標的的手續費佔總資產組合價值的比例，總和後才為真正的手續費比例
            commission_fee = commission_series.sum()

        return commission_fee

    # 載入benchmark作為績效對照，與計算策略表現指標
    # benchmark可為大盤或個股，也可為另一策略
    def _load_benchmark_data(self, benchmark):
        # Case1:待改：若無設定benchmark，即以策略本身為benchmark（以因應部分指標需benchmark計算）
        if benchmark == "None":
            self.benchmark_data_dict["percent"] = self.sum_percent_series
        
        # Case2: Benchmark為str，代表為某實體標的(e.g: "SPY")
        elif isinstance(benchmark, str):
            benchmark_adjclose = self.db.get_universe_df([benchmark], data_type="adjclose", data_format="all")
            adjclose_percent_series = benchmark_adjclose.squeeze().pct_change().fillna(0)
            #對照資產組合的時間序列，抓取對應的benchmark序列
            self.benchmark_data_dict["percent"] = adjclose_percent_series[self.sum_percent_series.index]
            
            
        #待改：若benchmark為Backtest類（其他策略）
        else:
            pass
        self.benchmark_data_dict["value"] = (1+self.benchmark_data_dict["percent"]).cumprod() * self.options["initial_value"]
    pass

    #依序啟動回測過程中各個函數，完成回測績效計算
    def activate(self):
        start_time = time.time()
        self._decay_weight_df()
        self._cal_return()
        self._load_benchmark_data(self.options["benchmark"])
        end_time = time.time()
        logging.info("回測共耗費{:.3f}秒\n".format(end_time - start_time))

    def show_figure(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()           # 做镜像处理

        weight_sum_series = self.weight_df.sum(axis=1)
        # x軸：日期
        x = self.sum_value_series.index
        y1 = self.sum_value_series.values
        #y3 = weight_sum_series[weight_sum_series.index == x].values
        ax1.plot(x, y1)
        if self.options["benchmark"] != "None":
            # y2: benchmark績效
            y2 = self.benchmark_data_dict["value"]
            ax1.plot(x, y2)
        # y1: 策略績效
        
        
        ax1.set_xlabel("Date")    #设置x轴标题
        # ax1.set_ylabel("Value", color='b')   #设置Y1轴标题
        # ax2.set_ylabel("Weight",color ='g')   #设置Y2轴标题
        
        plt.show()

    #評估策略表現，計算各類績效指標，並以self.performance此Dictionary儲存
    def evaluate(self, show=True):
        avg_fee = cal_avg_commision(self.performance["Commission Fee Series"], freq=self.options["freq"])
        avg_return, avg_volatility, avg_sharpe = cal_basic_performance(self.sum_percent_series)
        MDD, date_interval = cal_maxdrawdown(self.sum_value_series)
        # de_extreme_ratio_df, ratio_df = cal_profit_ratio(self.percent_df)
        
        win_rate = cal_win_rate(self.sum_percent_series, self.benchmark_data_dict["percent"])
        performance_per_Yr_df = cal_performance_per_yr(self.sum_percent_series, self.benchmark_data_dict["percent"])
        performance_per_Qr_df = cal_performance_per_quarter(self.sum_percent_series, self.benchmark_data_dict["percent"])
        
        self.performance["Average Annualized Return"] = avg_return
        self.performance["Average Annualized Volatility"] = avg_volatility
        self.performance["Average Sharpe"] = avg_sharpe
        self.performance["Calmar Ratio"] = avg_return / (MDD + 0.0001) #避免MDD為0時報錯
        self.performance["MDD"] = MDD
        self.performance["MDD INRERVAL"] = date_interval
        self.performance["WIN RATE"] = win_rate
        self.performance["Annualized Avg Commision Fee"] = avg_fee
        # self.performance["De_extreme Ratio"] = de_extreme_ratio_df
        # self.performance["Ratio"] = ratio_df
        self.performance["Performance Per Yr"] = performance_per_Yr_df
        self.performance["Performance Per Qr"] = performance_per_Qr_df

        if show == True:
            print("Annualized Return:", str(round(100*avg_return,2))+ "%")
            print("Annualized Volatility:", str(round(100*avg_volatility,2))+ "%")
            print("Sharpe:", round(avg_sharpe, 2))
            print("Calmar Ratio:", round(avg_return / (MDD + 0.0001), 2)) #避免MDD為0時報錯
            print("MDD INRERVAL:", datetime.strftime(date_interval[0], "%Y-%m-%d"),'~',
                                   datetime.strftime(date_interval[1], "%Y-%m-%d"))
            print("MDD:", MDD)
            print("WIN RATE:", win_rate)
            print("Annualized Avg Commision Fee:", round(100*avg_fee, 2),'%')
            print("\nPerformance Per Year:\n", performance_per_Yr_df)
            print("\nPerformance Per Quarter:\n", performance_per_Qr_df)
            print("\nPortfolio Value:\n", self.sum_value_series)
            
    #將回測紀錄存於以策略命名的資料夾中，資料夾結構（Universe_name/回測區段日期）
    def save_record(self): 
        #建立回測數據的資料夾名稱
        universe_filePath = os.path.join("Record",  self.options["universe_name"])
        start_date_str = self.options["start_date"].strftime("%Y-%m-%d")
        end_date_str = self.options["end_date"].strftime("%Y-%m-%d")
        date_filePath = os.path.join(universe_filePath, start_date_str+'_'+end_date_str)
        
        folderPath = date_filePath
        makeFolder(folderPath)

        #以當前時間戳記為檔名，儲存成Excel檔  
        current_time_str = time.strftime("%Y-%m-%d_%I-%M-%S", time.localtime())
        Performance_fileName = os.path.join(folderPath, current_time_str+".xlsx")
        Performance_excel = pd.ExcelWriter(Performance_fileName, engine='xlsxwriter')   # Creating Excel Writer Object from Pandas  
        
        (self.sum_value_series).to_excel(Performance_excel, sheet_name='Sum_Value')
        (self.weight_df).to_excel(Performance_excel, sheet_name='Weight')
        (self.performance["Performance Per Yr"]).to_excel(Performance_excel, sheet_name='Per Yr')
        (self.performance["Performance Per Qr"]).to_excel(Performance_excel, sheet_name='Per Qr')

        #將績效評估為單一數值的指標，另外以「Evaluation」工作頁存放
        item_for_log_list = [
            "Average Annualized Return", 
            "Average Annualized Volatility", 
            "Average Sharpe",
            "WIN RATE",
            "MDD",
            "MDD INRERVAL",
            "Annualized Avg Commision Fee",
            ]
        
        single_value_performance_dict = {key: self.performance[key] for key in item_for_log_list}
        performance_df = pd.DataFrame(list(single_value_performance_dict.items()), columns=['Item', 'Value'])
        performance_df.to_excel(Performance_excel, sheet_name='Evaluation')
        
        hyperParameters_df = pd.DataFrame(list(self.options["hyperParameters_dict"].items()), columns=['Item', 'Value'])
        hyperParameters_df.to_excel(Performance_excel, sheet_name='hyperParameters')
        
        Performance_excel.save()
        