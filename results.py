from typing import Final

from colorama import Fore, Style
import pandas as pd

from config import Configuration
from enums import RebalanceFrequency
from models import ModelType

class EvaluationResults:
	DAYS_PER_YEAR: Final[float] = 365.25

	symbol: str
	buy_and_hold_performance: float
	r2_score_training: float
	r2_score_validation: float
	mean_absolute_error_training: float
	mean_absolute_error_validation: float
	long_cash: float
	short_cash: float
	all_cash: float
	all_trades: int
	long_trades: int
	short_trades: int
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
			rebalance_frequency:
			RebalanceFrequency,
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
		self.all_trades = 0
		self.long_trades = 0
		self.short_trades = 0
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

	def submit_trade(self, returns: float, long: bool) -> None:
		if long:
			self.long_cash += returns
			self.long_cash -= self.slippage
			self.all_cash += returns
			self.long_trades += 1
		else:
			self.short_cash -= returns
			self.short_cash -= self.slippage
			self.all_cash -= returns
			self.short_trades += 1
		self.all_trades += 1
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
		print(f"{prefix} Model performance (long): {get_performance_trade_string(long_performance, self.long_trades)}")
		print(f"{prefix} Model performance (short): {get_performance_trade_string(short_performance, self.short_trades)}")
		print(f"{prefix} Model performance (all): {get_performance_trade_string(total_performance, self.all_trades)}")
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