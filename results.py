from typing import Final
from statistics import stdev, mean
from math import sqrt

from colorama import Fore, Style
import pandas as pd

from config import Configuration
from data import TrainingData
from enums import RebalanceFrequency
from models import ModelType
from technical import DAYS_SINCE_X

class EvaluationResults:
	DAYS_PER_YEAR: Final[float] = 365.25
	TRADING_DAYS_PER_YEAR: Final[float] = 252

	symbol: str
	buy_and_hold_performance: float
	r2_score_training: float
	r2_score_validation: float
	mean_absolute_error_training: float
	mean_absolute_error_validation: float
	long_cash: float
	short_cash: float
	all_cash: float
	long_trades: list[tuple[float, float]]
	short_trades: list[tuple[float, float]]
	all_trades: list[tuple[float, float]]
	slippage: float
	start: pd.Timestamp
	end: pd.Timestamp
	rebalance_frequency: RebalanceFrequency
	model_name: str
	model_type: ModelType
	parameters: dict[str, int | str]
	quantiles: list[float] | None
	result_category_id: int | None
	result_category: str | None

	def __init__(
		self,
		symbol: str,
		buy_and_hold_performance: float,
		r2_score_training: float,
		r2_score_validation: float,
		mean_absolute_error_training: float,
		mean_absolute_error_validation: float,
		slippage: float,
		start: pd.Timestamp,
		end: pd.Timestamp,
		rebalance_frequency: RebalanceFrequency,
		model_name: str,
		model_type: ModelType,
		parameters: dict[str, int | str],
		result_category_id: int | None,
		result_category: str | None
	):
		assert slippage >= 0
		assert start < end
		self.symbol = symbol
		self.buy_and_hold_performance = buy_and_hold_performance
		self.r2_score_training = r2_score_training
		self.r2_score_validation = r2_score_validation
		self.mean_absolute_error_training = mean_absolute_error_training
		self.mean_absolute_error_validation = mean_absolute_error_validation
		self.all_cash = Configuration.INITIAL_CASH
		self.long_cash = Configuration.INITIAL_CASH
		self.short_cash = Configuration.INITIAL_CASH
		self.all_trades = []
		self.long_trades = []
		self.short_trades = []
		self.slippage = slippage
		self.start = start
		self.end = end
		self.rebalance_frequency = rebalance_frequency
		self.model_name = model_name
		self.model_type = model_type
		self.parameters = parameters
		self.quantiles = None
		self.result_category_id = result_category_id
		self.result_category = result_category

	def submit_trade(
		self,
		long: bool,
		absolute_return: float,
		relative_return: float,
		risk_free_rate: float,
		time: pd.Timestamp
	) -> None:
		if long:
			self.long_cash += absolute_return
			self.long_cash -= self.slippage
			self.all_cash += absolute_return
			self.long_trades.append((relative_return, risk_free_rate))
			self.all_trades.append((relative_return, risk_free_rate))
			if Configuration.SHOW_TRADES:
				print(f"{time.date()} Long trade: {self.format_percentage(relative_return)}")
		else:
			self.short_cash -= absolute_return
			self.short_cash -= self.slippage
			self.all_cash -= absolute_return
			relative_short_return = 1 / (relative_return + 1) - 1
			self.short_trades.append((relative_short_return, risk_free_rate))
			self.all_trades.append((relative_short_return, risk_free_rate))
			if Configuration.SHOW_TRADES:
				print(f"{time.date()} Short trade: {self.format_percentage(relative_return)}")
		self.all_cash -= self.slippage

	def get_model_name(self) -> str:
		strings = [str(x) for x in self.parameters.values()]
		arguments = ", ".join(strings)
		return f"{self.model_name}({arguments})"

	def print_stats(self, symbol: str) -> None:
		def get_performance_trade_string(performance: float, trades: int) -> str:
			performance_string = self.get_performance_string(performance)
			if trades == 0:
				return f"-"
			elif trades == 1:
				return f"{performance_string} (1 trade)"
			else:
				return f"{performance_string} ({trades} trades)"

		buy_and_hold_performance = self._get_annualized_performance(self.buy_and_hold_performance)
		long_performance = self.get_annualized_long_performance()
		short_performance = self.get_annualized_short_performance()
		total_performance = self.get_annualized_performance()
		model_name = self.get_model_name()
		prefix = f"[{symbol} {model_name}]"
		print(f"{prefix} R2 scores: training {self.r2_score_training:.3f}, validation {self.r2_score_validation:.3f}")
		print(f"{prefix} Mean absolute error: {self.mean_absolute_error_training:.4f} training, {self.mean_absolute_error_validation:.4f} validation")
		print(f"{prefix} Buy and hold performance: {self.get_performance_string(buy_and_hold_performance)}")
		print(f"{prefix} Model performance (long): {get_performance_trade_string(long_performance, len(self.long_trades))}")
		print(f"{prefix} Model performance (short): {get_performance_trade_string(short_performance, len(self.short_trades))}")
		print(f"{prefix} Model performance (all): {get_performance_trade_string(total_performance, len(self.all_trades))}")
		print(f"{prefix} Signal/return quantiles: {self.get_quantile_string(self.quantiles)}")

	@staticmethod
	def get_quantile_string(quantiles: list[float]) -> str:
		quantile_string = ", ".join([EvaluationResults.format_percentage(x) for x in quantiles])
		quantile_delta = EvaluationResults.get_quantile_delta(quantiles)
		return f"{quantile_string} ({EvaluationResults.format_percentage(quantile_delta)})"

	@staticmethod
	def get_quantile_cells(quantiles: list[float]) -> list[str]:
		quantile_delta = EvaluationResults.get_quantile_delta(quantiles)
		cells = [EvaluationResults.format_percentage(x) for x in quantiles + [quantile_delta]]
		return cells

	@staticmethod
	def get_quantile_delta(quantiles: list[float]) -> float:
		quantile_delta = abs(quantiles[-1] - quantiles[0])
		return quantile_delta

	def get_annualized_long_performance(self) -> float:
		performance = self._get_cash_performance(self.long_cash)
		annualized_performance = self._get_annualized_performance(performance)
		return annualized_performance

	def get_annualized_short_performance(self) -> float:
		performance = self._get_cash_performance(self.short_cash)
		annualized_performance = self._get_annualized_performance(performance)
		return annualized_performance

	def get_annualized_performance(self) -> float:
		performance = self._get_cash_performance(self.all_cash)
		annualized_performance = self._get_annualized_performance(performance)
		return annualized_performance

	def get_long_sharpe_ratio(self) -> float:
		return self._get_sharpe_ratio(self.long_trades)

	def get_short_sharpe_ratio(self) -> float:
		return self._get_sharpe_ratio(self.short_trades)

	def get_total_sharpe_ratio(self) -> float:
		return self._get_sharpe_ratio(self.all_trades)

	@staticmethod
	def get_performance_string(performance: float) -> str:
		return EvaluationResults.format_percentage(performance - 1)

	@staticmethod
	def format_percentage(percentage: float) -> str:
		output = f"{percentage:+.2%}"
		if percentage > 0:
			output = f"{Fore.GREEN}{output}{Style.RESET_ALL}"
		elif percentage < 0:
			output = f"{Fore.RED}{output}{Style.RESET_ALL}"
		return output

	@staticmethod
	def _get_cash_performance(cash: float) -> float:
		performance = cash / Configuration.INITIAL_CASH
		return performance

	def _get_annualized_performance(self, performance):
		if performance <= 0:
			return 0
		days = (self.end - self.start).days
		annualized_performance = performance**(self.DAYS_PER_YEAR / days)
		return annualized_performance

	def _get_sharpe_ratio(self, returns_data: list[tuple[float, float]]) -> float | None:
		if len(returns_data) < 10:
			return None
		match self.rebalance_frequency:
			case RebalanceFrequency.DAILY | RebalanceFrequency.DAILY_SPLIT:
				factor = self.TRADING_DAYS_PER_YEAR
			case RebalanceFrequency.WEEKLY:
				factor = self.DAYS_PER_YEAR / 7
			case RebalanceFrequency.MONTHLY:
				factor = 12
			case _:
				raise Exception("Unknown rebalance frequency")
		returns = [returns for returns, _risk_free_rate in returns_data]
		risk_free_returns = [risk_free_rate for _returns, risk_free_rate in returns_data]
		ratio = (factor * mean(returns) - mean(risk_free_returns)) / stdev(returns)
		annualized_sharpe_ratio = ratio / sqrt(EvaluationResults.TRADING_DAYS_PER_YEAR)
		return annualized_sharpe_ratio