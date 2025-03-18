from collections import defaultdict
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, f1_score as get_f1_score, confusion_matrix

from enums import FeatureCategory, RebalanceFrequency
from config import Configuration

class EvaluationResults:
	DAYS_PER_YEAR: Final[float] = 365.25
	TRADING_DAYS_PER_YEAR: Final[int] = 252
	DAYS_PER_WEEK: Final[int] = 7
	MONTHS_PER_YEAR: Final[int] = 12

	buy_and_hold_performance: float
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

	def __init__(self, buy_and_hold_performance: float, slippage: float, start: pd.Timestamp, end: pd.Timestamp, rebalance_frequency: RebalanceFrequency):
		assert slippage >= 0
		assert start < end
		self.buy_and_hold_performance = buy_and_hold_performance
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

	def print_stats(self, symbol: str, model_name: str) -> None:
		days = (self.end - self.start).days
		buy_and_hold_performance = self.buy_and_hold_performance**(self.DAYS_PER_YEAR / float(days))
		long_performance = self.get_annualized_long_performance()
		short_performance = self.get_annualized_short_performance()
		total_performance = self.get_annualized_performance()
		print(f"[{symbol} {model_name}] Buy and hold performance: {self.get_performance_string(buy_and_hold_performance)}")
		print(f"[{symbol} {model_name}] Model performance (long): {self.get_performance_string(long_performance)}")
		print(f"[{symbol} {model_name}] Model performance (short): {self.get_performance_string(short_performance)}")
		print(f"[{symbol} {model_name}] Model performance (all): {self.get_performance_string(total_performance)}")

	def get_annualized_long_performance(self):
		performance = self._get_cash_performance(self.long_cash, self.long_trades)
		annualized_performance = self._get_annualized_performance(performance)
		return annualized_performance

	def get_annualized_short_performance(self):
		performance = self._get_cash_performance(self.short_cash, self.short_trades)
		annualized_performance = self._get_annualized_performance(performance)
		return annualized_performance

	def get_annualized_performance(self):
		performance = self._get_cash_performance(self.all_cash, self.all_trades)
		annualized_performance = self._get_annualized_performance(performance)
		return annualized_performance

	@staticmethod
	def get_performance_string(performance: float) -> str:
		return f"{performance - 1:+.2%}"

	@staticmethod
	def _get_cash_performance(cash: float, trades: int | None = None) -> float:
		performance = cash / Configuration.INITIAL_CASH - 1
		if trades is not None and trades > 0 and Configuration.DAILY_VALIDATION:
			performance /= trades
		performance += 1
		return performance

	def _get_annualized_performance(self, performance):
		match self.rebalance_frequency:
			case RebalanceFrequency.DAILY:
				annualized_performance = performance ** self.TRADING_DAYS_PER_YEAR
			case RebalanceFrequency.WEEKLY:
				annualized_performance = performance ** (self.DAYS_PER_YEAR / self.DAYS_PER_WEEK)
			case RebalanceFrequency.MONTHLY:
				annualized_performance = performance ** self.MONTHS_PER_YEAR
			case _:
				raise Exception("Unknown rebalance frequency")
		return annualized_performance

class TrainingResults:
	precision_values: list[float]
	roc_auc_values: list[float]
	f1_scores: list[float]
	max_precision: float | None
	max_f1_score: float | None
	max_roc_auc: float | None
	best_model_parameters: dict[str, int] | None
	best_model: Any | None
	best_model_precision: float | None
	best_model_roc_auc: float | None
	best_model_f1_score: float | None
	best_model_confusion_matrix: Any | None
	best_model_x_training: pd.DataFrame | None
	best_model_y_training: pd.DataFrame | None
	best_model_x_validation: pd.DataFrame | None
	best_model_y_validation: pd.DataFrame | None
	best_model_predictions: Any | None
	best_feature_categories: frozenset[FeatureCategory] | None
	parameter_scores: defaultdict[str, defaultdict[int, list[tuple[float, float]]]]
	feature_category_f1_scores: defaultdict[frozenset[FeatureCategory], list[float]]

	def __init__(self):
		self.precision_values = []
		self.roc_auc_values = []
		self.f1_scores = []
		self.max_precision = None
		self.max_roc_auc = None
		self.max_f1_score = None
		self.best_model_parameters = None
		self.best_model = None
		self.best_model_precision = None
		self.best_model_roc_auc = None
		self.best_model_f1_score = None
		self.best_model_confusion_matrix = None
		self.best_model_x_training = None
		self.best_model_y_training = None
		self.best_model_x_validation = None
		self.best_model_y_validation = None
		self.best_model_predictions = None
		self.best_feature_categories = None
		self.parameter_scores = defaultdict(lambda: defaultdict(list))
		self.feature_category_f1_scores = defaultdict(list)

	def submit_model(self, x_training: pd.DataFrame, y_training: pd.DataFrame, x_validation: pd.DataFrame, y_validation: pd.DataFrame, model: Any, model_parameters: dict[str, int], feature_categories: frozenset[FeatureCategory]):
		predictions = model.predict(x_validation)
		predictions = (predictions > 0.5).astype(dtype=np.int8)
		precision = precision_score(y_validation, predictions)
		self.precision_values.append(precision)
		roc_auc = roc_auc_score(y_validation, predictions)
		self.roc_auc_values.append(roc_auc)
		f1_score = get_f1_score(y_validation, predictions)
		self.f1_scores.append(f1_score)

		if self.best_model is None or f1_score > self.best_model_f1_score:
			self.best_model_parameters = model_parameters
			self.best_model = model
			self.best_model_precision = precision
			self.best_model_roc_auc = roc_auc
			self.best_model_f1_score = f1_score
			self.best_model_confusion_matrix = confusion_matrix(y_validation, predictions)
			self.best_model_x_training = x_training
			self.best_model_y_training = y_training
			self.best_model_x_validation = x_validation
			self.best_model_y_validation = y_validation
			self.best_model_predictions = predictions
			self.best_feature_categories = feature_categories
		self.max_roc_auc = max(roc_auc, self.max_roc_auc if self.max_roc_auc is not None else roc_auc)
		self.max_f1_score = max(f1_score, self.max_f1_score if self.max_f1_score is not None else f1_score)
		self.max_precision = max(precision, self.max_precision if self.max_precision is not None else precision)
		positive_predictions = 0
		for label in predictions:
			if label == 1:
				positive_predictions += 1
		positive_prediction_rate = float(positive_predictions) / len(predictions)
		for name, value in model_parameters.items():
			self.parameter_scores[name][value].append((f1_score, positive_prediction_rate))
		self.feature_category_f1_scores[feature_categories].append(f1_score)