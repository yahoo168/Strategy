from .backest import *
import copy

class Strategy(object):
	def __init__(self, universe_ticker_list:list, options=dict()):
		# 讀取同層資料夾下的option.txt，取得預設的回測設置
		self.options = self._read_options()
		# 在策略頁面若有調整則更新回測選項
		self.options.update(options)
		# 建立資料庫
		self.db = Database(self.options["database_path"])
		# 調整原始設定的日期，改為最接近該日的下一個交易日（若該日即為交易日則不變）
		self.options["start_date"] = self.db.get_next_tradeDate(self.options["start_date"], shift_days=self.options["decay_days"], country=self.options["country"])
		self.options["end_date"] = self.db.get_next_tradeDate(self.options["end_date"], shift_days=self.options["decay_days"], country=self.options["country"])
		self.universe_name = self.options["universe_name"]
		self.universe_ticker_list = universe_ticker_list
		self.weight_df = pd.DataFrame()	
		# 待改: 改成外部讀入
		self.hyperParameters_dict = self.options["hyperParameters_dict"]
		# self.universe_data_df_dict = self._load_universe_data()
	
	def _read_options(self):
		options = dict()
		with open('options.txt') as f:
			lines = f.readlines()
			for line in lines:
				option_name = line.split(":")[0].strip()
				option_value = line.split(":")[1].strip()
				options[option_name] = option_value

		float_option_name_list = ["commission_rate", ] 
		int_option_name_list = ["initial_value", "decay_days"]

		for option_name in float_option_name_list:
			options[option_name] = float(options[option_name])

		for option_name in int_option_name_list:
			options[option_name] = int(options[option_name])

		return options

	# 載入標的開高低收資料
	# 待改：好像不會用到了
	def _load_universe_data(self):
		universe_data_df_dict = dict()
		universe_data_df_dict["adjclose"] = self.db.get_stock_data_df(item="open", target_ticker_list=self.universe_ticker_list, \
													start_date=self.options["start_date"], end_date=self.options["end_date"], country=self.options["country"])
		return universe_data_df_dict

	def get_universe_df(self, item, pre_fetch_nums=0):
		item_df = self.db.get_cache_df(asset_class="stock", universe_name=self.options["universe_name"], item=item, \
									   start_date=self.options["start_date"], end_date=self.options["end_date"], \
									   country=self.options["country"], pre_fetch_nums=pre_fetch_nums)
		
		if item_df.empty == True:
			item_df = self.db.get_stock_data_df(item=item, target_ticker_list=self.universe_ticker_list, \
										start_date=self.options["start_date"], end_date=self.options["end_date"], \
										country=self.options["country"], pre_fetch_nums=pre_fetch_nums)
		
			cache_folderPath = os.path.join(self.db.cache_folderPath, "stock", self.options["universe_name"])
			make_folder(cache_folderPath)
			start_date = self.options["start_date"] + timedelta(days = -pre_fetch_nums)
			fileName = item + '_' + datetime2str(start_date) + '_' + datetime2str(self.options["end_date"]) + ".csv"
			filePath = os.path.join(cache_folderPath, fileName)
			item_df.to_csv(filePath)

		return item_df
	
	def _hyperParameters_dict(self):
		pass
	
	#依照固定調倉頻率取得換倉日期，供策略調用（自動加入回測起始與結束日期）
	def _get_change_tradeDate_list(self, rebalance_freq):
		#待改：此部分應可引用：_get_backtest_period_tradeDate_list
		tradeDate_list = self.db.get_tradeDate_list(country=self.options["country"])
		start_date_index = tradeDate_list.index(self.options["start_date"])
		end_date_index = tradeDate_list.index(self.options["end_date"])
		Backtest_period_tradeDate_list = tradeDate_list[start_date_index:end_date_index+1]
		change_date_list = []

		#不調倉，買入即持有至回測結束
		if rebalance_freq=="none":
			pass

		#週調倉，每週一若為交易日則調倉，若遇國定休假則順延下一週
		elif rebalance_freq=="week":
			for date in Backtest_period_tradeDate_list:
				if date.weekday()==0:
					change_date_list.append(date)
				
		#月調倉，預設每月第一個交易日調參，以前後日的月份不同判斷
		elif rebalance_freq=="month":
			for i in range(len(Backtest_period_tradeDate_list)):
				if Backtest_period_tradeDate_list[i].month != Backtest_period_tradeDate_list[i-1].month:
					change_date_list.append(Backtest_period_tradeDate_list[i])

		#待改：季調
		elif rebalance_freq=="quarter":
			pass

		if self.options["start_date"] not in change_date_list:
			change_date_list.append(self.options["start_date"])

		if self.options["end_date"] not in change_date_list:
			change_date_list.append(self.options["end_date"])

		change_date_list.sort()
		return change_date_list

	#取得回測期間中的所有交易日
	def _get_backtest_period_tradeDate_list(self):
		tradeDate_list = self.db.get_tradeDate_list(country=self.options["country"])
		start_date_index = tradeDate_list.index(self.options["start_date"])
		end_date_index = tradeDate_list.index(self.options["end_date"])
		Backtest_period_tradeDate_list = tradeDate_list[start_date_index:end_date_index+1]
		return Backtest_period_tradeDate_list

	#檢查輸出的策略矩陣無異常
	def _check_weight_df(self, weight_df, options):
		#待改：暫時註解以容許short position
		#assert((weight_df.values >= 0).all()), "權重不可為負"
		assert((weight_df.sum(axis=1) <=1.001).all()), "權重總和不可大於1"
		assert(weight_df.index[0]>=options["start_date"]),"權重矩陣起始日期，不可早於回測期間起始日（已調整後的實際交易日）"
		assert(weight_df.index[-1]<=options["end_date"]), "權重矩陣結束日期，不可晚於回測期間結束日（已調整後的實際交易日）"
	
	#計算回測損益與評估績效，並依選項儲存回測紀錄與顯示績效圖表
	def cal_pnl(self, show_evaluation=True, show_figure=True, save_record=True, generate_report=False):
		#取得策略權重矩陣
		weight_df = self.cal_weight()
		#檢查策略輸出的權重矩陣是否符合基本規範
		self._check_weight_df(weight_df, self.options)
		#使用權重矩陣建立回測物件
		backtest = Backtest(weight_df, self.db, self.options)
		#計算回測損益
		backtest.activate()
		#評估績效
		backtest.evaluate(show=show_evaluation)
		#儲存回測紀錄
		if save_record == True:
			backtest.save_record()
		#顯示績效圖表
		if show_figure == True:
			backtest.show_figure()
		if generate_report == True:
			backtest.generate_report()

		return backtest
	
	#產生包含所有試驗參數組合的字典
	def _generate_parameters(self, testPara_name_list, low_bound, high_bound, times):
		#因計算笛卡爾積時字典的值須為列表，此處先將原數值轉為元素數量為1的列表
		#Key:參數名稱，value:list(該參數將測試的各種可能值)
		p_range_dict = {k: [v] for k, v in self.options["hyperParameters_dict"].items()}
		
		#逐一調整各參數
		for para_name in testPara_name_list:
			#依據設定的倍數高低值與試驗次數多，求出試驗參數相較原參數的變化幅度
			
			p_list = p_range_dict[para_name]*(1+np.linspace(low_bound, high_bound, times))   
			#若原參數為整數，須將調整後的試驗參數也四捨五入為整數，以避免部分參數不得使用小數
			if isinstance(p_range_dict[para_name], int):
				for i in range(len(p_list)):
					p_list[i] = round(p_list[i])

			#將小數點後帶0的整數，整理為整數，如2.0變成2
			p_range_dict[para_name] = p_list.astype(int)
		
		#回傳完整的試驗參數字典
		return p_range_dict
	
	#找出Sharpe最大的一組參數與其Sharpe值，並回傳試驗參數字典的試驗結果
	def parameter_opmitize(self, testPara_name_list, low_bound = -1, high_bound=1, times=5, if_show_figure=False):
		from itertools import product

		#生成待測試的參數組合
		p_range_dict = self._generate_parameters(testPara_name_list, low_bound=low_bound, high_bound=high_bound, times=times)
		
		#利用試驗參數字典取得笛卡爾積
		parameters_dict_list = list(dict(zip(p_range_dict.keys(), values)) for values in product(*p_range_dict.values()))
		
		#取得策略矩陣
		weight_df = self.cal_weight()
		#檢查輸出的策略矩陣以防錯
		self._check_weight_df(weight_df, self.options)

		#依據各種參數組合，逐次進行回測
		for index, parameters_dict in enumerate(parameters_dict_list, 1):
			percentage = 100*index/len(parameters_dict_list)
			sys.stdout.write('\r'+"敏感度測試：完成度{percentage:.2f}%\n".format(percentage=percentage))
		
			self.options["hyperParameters_dict"] = parameters_dict
			weight_df = self.cal_weight()
			self._check_weight_df(weight_df, self.options)
			backtest = Backtest(weight_df, self.options)
			backtest.activate()
			backtest.evaluate(show=False)
			average_sharpe = backtest.performance["Average Sharpe"]
		
			#在笛卡兒積參數字典多加平均夏普比率的key，以紀錄該組試驗參數的績效表現
			parameters_dict["Average Sharpe"] = average_sharpe

			print("Parameters:\n", parameters_dict)
			print("Sharpe:", backtest.performance["Average Sharpe"])
			print()

			#紀錄最佳夏普，預設為0
			max_sharpe = 0
		
			#紀錄績效表現最佳的參數組合
			argmax_dict = dict()

			#對比績效是否超過前高
			if backtest.performance["Average Sharpe"] > max_sharpe:
				max_sharpe = backtest.performance["Average Sharpe"]
				argmax_dict = parameters_dict
		
		import statistics
		
		#計算參數敏感度
		for para_name, p_range_list in p_range_dict.items():
			if para_name in testPara_name_list:
				#將其餘參數所有的可能值與此參數的定值搭配，取其Sharpe中位數，作為此參數為某定值時的Sharpe
				#待改：有沒有比較好的寫法？雖然目前不影響速度
				median_sharpe_list = list()
				for p_value in p_range_list:
					sharpe_list = list()
					for outcome in parameters_dict_list: 
						if outcome[para_name] == p_value:
							sharpe_list.append(outcome["Average Sharpe"])
					
					median_sharpe_list.append(statistics.median(sharpe_list))
			
				#繪製參數敏感性折線圖
				if if_show_figure:
					print("參數名稱:", para_name)
					print("參數變動範圍:", list(p_range_list))
					print("Sharpe中位數:", median_sharpe_list)
					
					print(median_sharpe_list)
					print(p_range_list)
					median_sharpe_series = pd.Series(median_sharpe_list, index=p_range_list)
					median_sharpe_series.plot(title="Sensitivity Analysis", grid=True)

					plt.xlabel(para_name)
					plt.ylabel('Sharpe Ratio')
					plt.show()
		
		print("參數最佳化可達成的最大Sharpe Ratio為{max_sharpe}，其參數組合為: ".format(max_sharpe=max_sharpe), argmax_dict)
		return max_sharpe, argmax_dict
		
# 每年重新執行的策略績效
# perfomance_per_yr_df = pd.DataFrame(index=range(2008, 2020), columns=["Return", "Std", "Sharpe", "MaxDrawdown"])
# for yr in range(2008, 2021):
#     op = options.copy()
#     start_date = str(yr)+"-01-01"
#     end_date = str(yr)+"-12-18"
#     op["start_date"] = start_date
#     op["end_date"] = end_date
#     ARC = Backtest(ARC_basic, universe_ticker_list, op)
#     ARC.activate()
    
#     start_value = end_value = ARC.sum_value_series[0]
#     end_value = ARC.sum_value_series[-1]
#     yr_return = (end_value-start_value) / start_value
#     yr_std = ARC.sum_value_series.pct_change().std()*sqrt(252)
#     yr_sharpe = yr_return / yr_std
#     MDD, date_interval = cal_maxdrawdown(ARC.sum_value_series)
#     perfomance_per_yr_df.loc[yr, "Return"] = yr_return
#     perfomance_per_yr_df.loc[yr, "Std"] = yr_std
#     perfomance_per_yr_df.loc[yr, "Sharpe"] = yr_sharpe
#     perfomance_per_yr_df.loc[yr, "MaxDrawdown"] = MDD

# print(perfomance_per_yr_df)
# perfomance_per_yr_df.to_excel("yr_perfomance_2020.xlsx")