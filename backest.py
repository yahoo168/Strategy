#from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json, pickle
import time
from self_finance_database_tool import *
from .evaluation import *
from .utils import *
from .report import *

class Backtest(object):
    """
    待補充：
    Backtest為回測物件，必要的輸入值為策略權重矩陣，與回測設定選項（options）
    物件方法概覽：
        & activate: 呼叫策略以取得權重df，並計算此權重在過去的每日價值變動
        & evaluate: 依照計算完成的每日百分比變動計算績效指標
        & save_record: 將各績效指標以及策略參數存檔
    """
    def __init__(self, weight_df:pd.DataFrame, db:Database, options:dict):
        self.weight_df = weight_df
        self.options = options
        
        # 透過路徑建立資料庫物件，以取用資料
        # self.db = Database(self.options["database_path"])
        # 資料庫改自外部接入
        self.db = db
        
        # 調整原始設定的日期，改為最接近該日的下一個交易日（若該日即為交易日則不變）
        self.start_date = self.options["start_date"]
        self.end_date = self.options["end_date"]

        #存放Benchmark的各類資料，便於績效比對。
        self.benchmark_data_dict = dict()

        #存放投資組合績效評估的各類資料，便於績效比對。
        self.performance = dict()
   
        #紀錄每次換倉的手續費序列
        self.performance["commission_fee_series"] = pd.Series(dtype="float64")

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
        # 回測最後一日若沒有策略指定權重，則以權重矩陣的最後一次指定權重作為最終權重
        # Note: 會多扣一次手續費，但計算績效較方便
        if self.end_date not in self.weight_df.index:
            logging.info("回測期間最後一日({date1})的權重未指定，故以策略最後指定權重之日({date2})為準以自動補足"
                    .format(date1=self.end_date.strftime("%Y-%m-%d"), date2=self.weight_df.index[-1].strftime("%Y-%m-%d")))

            # 複製最後一個row，更改其日期與權重
            self.weight_df = self.weight_df.append(self.weight_df.iloc[-1,:])
            date_index_list = self.weight_df.index.tolist()
            # 將其日期改為回測最後一日
            date_index_list[-1] = self.end_date
            self.weight_df.index = pd.Index(date_index_list)
        
        changeDate_list = self.weight_df.index
        holder_ticker_list = list(self.weight_df.columns)
        
        #取得持有標的列表，並呼叫資料表，以此取得調整價
        
        #self.price_df = self.db.get_stock_data_df(item="close", target_ticker_list=holder_ticker_list, \
        #                                            start_date=self.options["start_date"], end_date=self.options["end_date"], country=self.options["country"])
        
        # #待改：能否結合此處與strategy的cache機制？
        item = "close"
        price_df = self.db.get_cache_df(asset_class="stock", universe_name=self.options["universe_name"], item=item, \
                                       start_date=self.options["start_date"], end_date=self.options["end_date"], \
                                       country=self.options["country"])
        
        if price_df.empty == True:
            price_df = self.db.get_stock_data_df(item=item, target_ticker_list=holder_ticker_list, \
                                        start_date=self.options["start_date"], end_date=self.options["end_date"], \
                                        country=self.options["country"])
        
            cache_folderPath = os.path.join(self.db.cache_folderPath, "stock", self.options["universe_name"])
            fileName = item + '_' + datetime2str(self.options["start_date"]) + '_' + datetime2str(self.options["end_date"]) + ".csv"
            filePath = os.path.join(cache_folderPath, fileName)
            price_df.to_csv(filePath)

        self.price_df = price_df


        # 因應部分回測每日皆給定權重，可不進行矩陣分拆，以加速計算
        # 待改：上面的權重矩陣調整會影響此部分(因在回測期末多加一日權重，導致長度不一)
        pct_change_df = self.price_df.pct_change()
        if len(pct_change_df) == len(self.weight_df):
            self.sum_percent_series = (pct_change_df * self.weight_df).sum(axis=1)
            weight_diff_df = self.weight_df.shift(1) - self.weight_df
            #扣除交易手續費
            commission_fee_series = (abs(weight_diff_df).sum(axis=1) * self.options["commission_rate"]/2).fillna(0)
            #透過百分比變動計算資產組合的日報酬，資產初始金額由options給定
            self.sum_percent_series = self.sum_percent_series - commission_fee_series
            self.sum_value_series = (self.sum_percent_series+1).cumprod() * self.options["initial_value"]
            self.performance["Commission Fee Series"] = commission_fee_series
        
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
                #在計算percent_series已在最後一天扣除手續費，此處只是另外獨立紀錄
                interval_sum_percent_series, interval_commission_fee = self._cal_interval_return(interval)
                interval_sum_percent_series_list.append(interval_sum_percent_series)
                interval_commission_fee_list.append(interval_commission_fee)
            print()
            
            # 待改：為何不直接放進self.performance
            # 合成各換倉區間回測所得的百分比變動序列
            self.sum_percent_series = pd.concat(interval_sum_percent_series_list)
            self.performance["Commission Fee Series"] = pd.Series(interval_commission_fee_list, index=changeDate_list[1:])

            #透過百分比變動計算資產組合的日報酬，資產初始金額由options給定
            self.sum_value_series = (self.sum_percent_series+1).cumprod() * self.options["initial_value"]    
  
    # 距離期初日期與結束日期，各自最近的下一個交易日，也應在權重矩陣的調倉日中
    # 計算換倉區間的績效變動，換倉日前一日收盤後換倉，隔日為新倉位。
    # 回傳該區間的每日資產價值百分比變動（pd.Series）以及期末換倉日的手續費(float)
    def _cal_interval_return(self, interval):
        # 從給定的日期區間（interval）讀取此區間期初與結束的原始日期
        s_date, e_date = interval[0], interval[1]
        # 將期初日期與結束日期各往前調一個交易日，以實現換倉日當天開盤前即換完倉
        # 意即回測區間的第一日，便應計算該日的漲跌幅，而最後一日不計算（由下一個區間段處理）
        s_date_last_tradeDate = self.db.get_last_tradeDate(s_date-timedelta(days=1), country=self.options["country"])
        e_date_last_tradeDate = self.db.get_last_tradeDate(e_date-timedelta(days=1), country=self.options["country"])
        
        # 取得期初日（回測開始日 或初始換倉日）的目標權重
        s_weight = self.weight_df.loc[s_date,:]
        # 取得結束日（下一次換倉日）的目標權重，以便計算調倉手續費
        next_weight = self.weight_df.loc[e_date,:]
    
        # 截取該區間的價格df（往起始日前多取一天，以取得起始日當日的百分比變動）
        # 待改：用groupby datetime實現？
        mask = (self.price_df.index >= s_date_last_tradeDate) & (self.price_df.index <= e_date_last_tradeDate)
        percent_df = self.price_df.pct_change()[mask]
        # 待改：好像不用：將起始日的前一日百分比變動由Nan改為0，以避免出錯
        # percent_df.fillna(0, inplace=True)
        
        holder_ticker_list = percent_df.columns        
        # 確保weighted_cumprod_df的第一列為1，用於比較基準
        # 待改：為何不直接weighted_cumprod_df.iloc[0,:]=1
        percent_df.iloc[0,:] = 0
        # 將各部位百分比取連乘，並乘上起始日的權重分配，得出本次換倉區間的加權累積報酬
        weighted_cumprod_df = (percent_df+1).cumprod() * s_weight
        
        # 此算法可兼顧多/空倉收益
        weighted_cum_return = weighted_cumprod_df - weighted_cumprod_df.iloc[0, :]
        sum_percent_series = (weighted_cum_return.sum(axis=1) + 1).pct_change()

        # 去除第一列資料，即換倉日前一交易日的百分比變動（原是為配合計算百分比變動而加入）
        sum_percent_series = sum_percent_series[1:]
        commission_fee = 0
        # 策略權重區間段內可能皆為假期，不存在任何報價會導致sum_percent_series為空
        if len(sum_percent_series) > 0:
            # 計算出換倉手續費後在最後一日扣除
            e_value = weighted_cumprod_df.iloc[-1,:]
            # 取得倉位持現比例，以便計算調倉手續費 (待改：此處用abs計算放空部位是否合適)
            cash_weight = max(1 - abs(s_weight).sum(), 0)
            commission_fee = self._cal_commision_fee(cash_weight, e_value, next_weight) 
            sum_percent_series[-1] -= commission_fee

        return sum_percent_series, commission_fee

    # 計算換倉交易成本（手續費)，以待調整權重佔總部位比例計算手續費
    def _cal_commision_fee(self, cash_weight, e_value, next_weight):
        commission_fee = 0
        if abs(e_value).sum() != 0:
            # e_value為加權後期末的各標的價值(基準值為1）
            # 計算本區間因價格變動所導致的權重偏移，可算出實際的期末權重(e_weight)
            e_weight = e_value / (abs(e_value).sum() + cash_weight)
            # 期末權重 - 目標權重 = 倉位須調整的部位佔整體組合比例，以此計算交易手續費佔總組合價值的比例
            # 因原始手續費0.4%，為買進賣出的總數，單向交易僅一半
            commission_series = abs(e_weight - next_weight) * (self.options["commission_rate"])/2
            # commission_series為各標的的手續費佔總資產組合價值的比例，總和後才為真正的手續費比例
            commission_fee = commission_series.sum()
 
        return commission_fee

    # 載入benchmark作為績效對照，與計算策略表現指標，benchmark可為大盤或個股，也可為另一策略
    def _load_benchmark_data(self, benchmark):
        # Case1:待改：若無設定benchmark，即以策略本身為benchmark（以因應部分指標需benchmark計算）
        if benchmark == "None":
            self.benchmark_data_dict["percent"] = self.sum_percent_series
        
        # Case2: Benchmark為str，代表為某實體標的(e.g: "SPY")
        elif isinstance(benchmark, str):
            benchmark_price = self.db.get_stock_data_df(item="close", target_ticker_list=[benchmark],
                                                           start_date=self.options["start_date"], end_date=self.options["end_date"], country=self.options["country"])
            
            cum_ = benchmark_price / benchmark_price.iloc[0,:]
            cum_ = cum_ - cum_.iloc[0,:]
            #print("benchmark_cumRet")
            #print(cum_)
            benchmark_percent_series = benchmark_price.squeeze().pct_change().fillna(0)
            #對照資產組合的時間序列，抓取對應的benchmark序列
            self.benchmark_data_dict["percent"] = benchmark_percent_series[self.sum_percent_series.index]
        #待改：若benchmark為Backtest類（其他策略）
        else:
            pass
        self.benchmark_data_dict["value"] = (1+self.benchmark_data_dict["percent"]).cumprod() * self.options["initial_value"]
    pass

    #依序啟動回測過程中各個函數，完成回測績效計算
    def activate(self, show=True):
        start_time = time.time()
        self._decay_weight_df()
        self._cal_return()
        self._load_benchmark_data(self.options["benchmark"])
        end_time = time.time()
        if show:
            logging.info("回測共耗費{:.3f}秒\n".format(end_time - start_time))

    def show_figure(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # 鏡像處理
        #ax2.set_ylim(, 0)
        #ax2.set_ylim(auto=True)
        weight_sum_series = self.weight_df.sum(axis=1)
        # x軸：日期
        x = self.sum_value_series.index
        y1 = self.sum_value_series.values
        ax1.plot(x, y1, label="Strategy")
        # y2: benchmark績效
        if self.options["benchmark"] != "None":
            y2 = self.benchmark_data_dict["value"]
            # y3: 累積超額績效
            y3 = y1 - y2
            ax2.plot(x, y3, label="Excess Return", color='rebeccapurple', linewidth=2)
            ax1.plot(x, y2, label="Benchmark")

        # y1: 策略績效
        ax1.set_xlabel("Date")    # 設置x轴標題
        ax1.set_ylabel("Value", color='black')   # 設置Y1軸標題
        # ax2.set_ylabel("Weight",color ='g')   # 設置Y2軸標題
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.show()

    #評估策略表現，計算各類績效指標，並以Dict(self.performance)儲存
    def evaluate(self, show=True):
        avg_fee = cal_avg_commision(self.performance["Commission Fee Series"], freq=self.options["freq"])
        avg_return, avg_volatility, avg_sharpe = cal_basic_performance(self.sum_percent_series)
        MDD, date_interval = cal_maxdrawdown(self.sum_value_series)
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
        self.performance["Performance Per Yr"] = performance_per_Yr_df
        self.performance["Performance Per Qr"] = performance_per_Qr_df

        # 分析持倉各指標
        self.weight_analysis_dict = analyze_weight_df(self, self.weight_df, rank_nums=10)

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
            print("Annualized Turnover Rate:", round(100*self.weight_analysis_dict["annualized_turnover_rate"],2),'%')
            #print("\nPerformance Per Year:\n", performance_per_Yr_df)
            #print("\nPerformance Per Quarter:\n", performance_per_Qr_df)
            print("\nPortfolio Value:\n", self.sum_value_series)
            
    # 將回測紀錄存於以策略命名的資料夾中，資料夾結構（Universe_name/回測區段日期）
    def save_record(self): 
        # 建立回測數據的資料夾名稱
        universe_filePath = os.path.join("Record",  self.options["universe_name"])
        start_date_str = self.options["start_date"].strftime("%Y-%m-%d")
        end_date_str = self.options["end_date"].strftime("%Y-%m-%d")
        date_filePath = os.path.join(universe_filePath, start_date_str+'_'+end_date_str)
        
        folderPath = date_filePath
        makeFolder(folderPath)

        # 生存回測數據Excel報表，以當前時間戳記為檔名
        current_time_str = time.strftime("%Y-%m-%d_%I-%M-%S", time.localtime())
        Performance_fileName = os.path.join(folderPath, current_time_str+".xlsx")
        Performance_excel = pd.ExcelWriter(Performance_fileName, engine='xlsxwriter')   # Creating Excel Writer Object from Pandas  
        
        (self.sum_value_series).to_excel(Performance_excel, sheet_name='Sum_Value')
        (self.weight_df).to_excel(Performance_excel, sheet_name='Weight')
        (self.performance["Performance Per Yr"]).to_excel(Performance_excel, sheet_name='Per Yr')
        (self.performance["Performance Per Qr"]).to_excel(Performance_excel, sheet_name='Per Qr')

        # 將績效評估為單一數值的指標，另外以「Evaluation」工作頁存放
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

    def generate_report(self):
        performance_dict = self.performance
        avg_return = round(100*performance_dict["Average Annualized Return"],2)
        avg_volatility = round(100*performance_dict["Average Annualized Volatility"],2)
        avg_sharpe = round(performance_dict["Average Sharpe"],2)
        MDD = round(100*performance_dict["MDD"],2)
        date_interval = performance_dict["MDD INRERVAL"]
        performance_per_Yr_df = performance_dict["Performance Per Yr"]
        performance_per_Qr_df = performance_dict["Performance Per Qr"]

        content = list()
        # 添加標題
        content.append(Graphs.draw_title('Analysis of Strategy: {}'.format(self.options["strategy_name"])))
        # 回測基本規格設定
        content.append(Graphs.draw_little_title("Backtesting Setting:"))
        content.append(Graphs.draw_text("- Universe Pool: {}".format(self.options["universe_name"])))
        
        start_date = datetime.strftime(self.options["start_date"], "%Y-%m-%d")
        end_date = datetime.strftime(self.options["end_date"], "%Y-%m-%d")
        content.append(Graphs.draw_text("- Perioid: {0} ~ {1}".format(start_date, end_date)))
        content.append(Graphs.draw_text("- Rebalance Frequence: {}".format(self.options["freq"])))
        content.append(Graphs.draw_text("- Commission Rate: {}%".format(100*self.options["commission_rate"])))
        content.append(Graphs.draw_text("- Country: {}".format(self.options["country"])))
        content.append(Graphs.draw_text("- Benchmark: {}".format(self.options["benchmark"])))
        
        # 回測基本績效指標
        content.append(Spacer(0, 1*cm))
        content.append(Graphs.draw_little_title("Basic Performance Indicators:"))

        content.append(Graphs.draw_text("- Annualized Return: {}%".format(avg_return)))
        content.append(Graphs.draw_text("- Annualized Vol: {}%".format(avg_volatility)))
        content.append(Graphs.draw_text("- Annualized Sharpe: {}".format(avg_sharpe)))
        content.append(Graphs.draw_text("- Max Drowdown: {}%".format(MDD)))
 
        # 生成累積報酬圖表
        #self.sum_value_series.plot(title="Cumulative Return").get_figure().savefig('temp1.png')
        
        df = pd.concat([self.sum_value_series.rename("Strategy"), self.benchmark_data_dict["value"]], axis=1)
        df.plot(title="Cumulative Strategy Return").get_figure().savefig('temp0.png')
        plt.close()
        img_0 = Image("temp0.png")
        #img_0.hAlign ='LEFT'

        cul_extra_return_series = self.sum_value_series / self.benchmark_data_dict["value"]
        cul_extra_return_series.plot(title="Cumulative Extra Return").get_figure().savefig('temp1.png')
        plt.close()
        img_1 = Image("temp1.png")
        #img_1.hAlign ='LEFT'

        img_0, img_1 = Image("temp0.png"), Image("temp1.png")   
        img_0.drawWidth, img_0.drawHeight = 10*cm, 8*cm
        img_1.drawWidth, img_1.drawHeight = 10*cm, 8*cm
        content.append(Table([[img_0, img_1]]))

        # 換頁（P2）
        content.append(PageBreak())
        # 分年績效
        content.append(Graphs.draw_little_title("Performance Per Year:"))
        content.append(Graphs.draw_table(performance_per_Yr_df, index_name="Date"))
        # 插入分隔符
        content.append(Spacer(0, 1.5*cm))
        # 分季績效
        content.append(Graphs.draw_little_title("Performance Per Quarter:"))
        content.append(Graphs.draw_table(performance_per_Qr_df, index_name="Date"))
        
        # 換頁（P3）
        content.append(PageBreak())
        content.append(Graphs.draw_title("Holding Analysis"))
        # 最大、最小、平均持有檔數
        content.append(Graphs.draw_text("- Max Holding Nums: {}".format(self.weight_analysis_dict["max_holding_nums"])))
        content.append(Graphs.draw_text("- Min Holding Nums: {}".format(self.weight_analysis_dict["min_holding_nums"])))
        content.append(Graphs.draw_text("- Mean Holding Nums: {}".format(round(self.weight_analysis_dict["mean_holding_nums"],2))))

        # 前10大平均持股權重（以整個回測區間計算）
        content.append(Graphs.draw_little_title("Average Weight Per Ticker(Largest 10):"))
        
        avg_weight_pet_ticker_table = Graphs.draw_table(self.weight_analysis_dict["avg_weight_per_ticker"], index_name="Ticker")
        
        # 前10大持股平均權重（橫截面計算取平均）
        N_rank_weight_df = self.weight_analysis_dict["N_rank_weight_series"].to_frame(name="Weight")
        N_rank_weight_df.plot(title="Rank(1-10) Average Weight").get_figure().savefig('temp2.png')
        plt.close()
        
        img_2 = Image("temp2.png")
        img_2.hAlign = "RIGHT"
        img_2.drawWidth, img_2.drawHeight = 10*cm, 8*cm
        content.append(Table([[avg_weight_pet_ticker_table, img_2]]))
        
        self.weight_analysis_dict["turnover_rate_series"].plot(title="Turnover Rate").get_figure().savefig('temp3.png')
        plt.close()
        self.weight_analysis_dict["weight_series"].plot(title="Portfolio Weight").get_figure().savefig('temp4.png')
        plt.close()
        
        img_3, img_4 = Image("temp3.png"), Image("temp4.png")        
        img_3.drawWidth, img_3.drawHeight = 10*cm, 8*cm
        img_4.drawWidth, img_4.drawHeight = 10*cm, 8*cm
        content.append(Table([[img_3, img_4]]))

        # 生成pdf文件
        doc = SimpleDocTemplate('report.pdf', pagesize=letter)
        doc.build(content)

        os.remove("temp0.png")
        os.remove("temp1.png")
        os.remove("temp2.png")
        os.remove("temp3.png")
        os.remove("temp4.png")