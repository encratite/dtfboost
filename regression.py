import warnings
from functools import partial
from itertools import product
from statistics import mean
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as get_mean_absolute_error
from sklearn.metrics import r2_score as get_r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from config import Configuration
from enums import RebalanceFrequency
from models import get_linear_models, get_random_forest_models, get_catboost_models, get_mlp_models, get_lightgbm_models, get_xgboost_models
from results import EvaluationResults
from wrapper import RegressionWrapper

def perform_regression(
		symbol: str,
		x_training: list[list[float]],
		y_training: list[float],
		x_validation: list[list[float]],
		y_validation: list[float],
		training_times: list[pd.Timestamp],
		validation_times: list[pd.Timestamp],
		deltas: list[float],
		rebalance_frequency: RebalanceFrequency,
		buy_and_hold_performance: float,
		process_id: int,
		process_count: int
	) -> list[EvaluationResults]:
	assets = pd.read_csv(Configuration.ASSETS_CONFIG)
	rows = assets[assets["symbol"] == symbol]
	if len(rows) == 0:
		raise Exception(f"No such symbol in assets configuration: {symbol}")
	asset = rows.iloc[0].to_dict()
	tick_size = asset["tick_size"]
	tick_value = asset["tick_value"]
	broker_fee = asset["broker_fee"]
	exchange_fee = asset["exchange_fee"]
	margin = asset["margin"]
	contracts = max(int(round(10000.0 / margin)), 1)
	slippage = 2 * contracts * (broker_fee + exchange_fee + Configuration.SPREAD_TICKS * tick_value)
	models = get_linear_models()
	if Configuration.MODEL_ENABLE_RANDOM_FOREST:
		models += get_random_forest_models()
	if Configuration.MODEL_ENABLE_CATBOOST:
		models += get_catboost_models()
	if Configuration.MODEL_ENABLE_LIGHTGBM:
		models += get_lightgbm_models()
	if Configuration.MODEL_ENABLE_XGBOOST:
		models += get_xgboost_models()
	if Configuration.MODEL_ENABLE_MLP:
		models += get_mlp_models()

	if Configuration.TRANSFORMER is not None:
		match Configuration.TRANSFORMER:
			case "StandardScaler":
				transformer = StandardScaler()
			case "RobustScaler":
				transformer = RobustScaler()
			case "QuantileTransformer":
				transformer = QuantileTransformer()
			case _:
				raise Exception("Unknown transformer specified")
		transformer.fit(x_training)
		x_training = transformer.transform(x_training)
		x_validation = transformer.transform(x_validation)

	warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
	warnings.filterwarnings("ignore", category=ConvergenceWarning)

	def filter_by_dayofweek(day: int, filter_time: pd.Timestamp) -> bool:
		return filter_time.dayofweek == day

	output = []
	tasks = []
	time_filters = [None]
	if rebalance_frequency == RebalanceFrequency.DAILY_SPLIT:
		days = [
			"Monday",
			"Tuesday",
			"Wednesday",
			"Thursday",
			"Friday",
		]
		time_filters = [(i, days[i], partial(filter_by_dayofweek, i)) for i in range(len(days))]
	task_id = None
	for model_name, model_type, model, parameters in models:
		if task_id is None:
			task_id = 0
		else:
			task_id += 1
		if task_id % process_count != process_id:
			continue
		tasks.append((model_name, model_type, model, parameters))
	tasks = [a + (b,) for a, b in product(tasks, time_filters)]

	if process_id == 0:
		wrapped_tasks = tqdm(tasks, desc="Evaluating models", colour="green")
	else:
		wrapped_tasks = tasks
	for model_name, model_type, model, parameters, time_filter in wrapped_tasks:
		result_category_id = None
		result_category = None
		x_training_filtered = x_training
		y_training_filtered = y_training
		x_validation_filtered = x_validation
		y_validation_filtered = y_validation
		if time_filter is not None:
			result_category_id, result_category, time_filter_function = time_filter
			if Configuration.SPLIT_TRAINING:
				x_training_filtered, y_training_filtered = filter_data_set(x_training, y_training, training_times, time_filter_function)
			x_validation_filtered, y_validation_filtered = filter_data_set(x_validation, y_validation, validation_times, time_filter_function)
		if isinstance(model, RegressionWrapper):
			model.set_validation(x_validation_filtered, y_validation_filtered)
		model.fit(x_training_filtered, y_training_filtered)
		training_predictions = model.predict(x_training_filtered)
		validation_predictions = model.predict(x_validation_filtered)
		r2_score_training = get_r2_score(y_training_filtered, training_predictions)
		r2_score_validation = get_r2_score(y_validation_filtered, validation_predictions)
		mean_absolute_error_training = get_mean_absolute_error(y_training_filtered, training_predictions)
		mean_absolute_error_validation = get_mean_absolute_error(y_validation_filtered, validation_predictions)
		evaluation_results = EvaluationResults(
			symbol,
			buy_and_hold_performance,
			r2_score_training,
			r2_score_validation,
			mean_absolute_error_training,
			mean_absolute_error_validation,
			slippage,
			validation_times[0],
			validation_times[-1],
			rebalance_frequency,
			model_name,
			model_type,
			parameters,
			result_category_id,
			result_category
		)
		last_trade_time: pd.Timestamp | None = None
		signal_returns: list[tuple[float, float]] = []
		signal_history = training_predictions.tolist()[-Configuration.SIGNAL_HISTORY:]
		for i in range(len(y_validation_filtered)):
			time = validation_times[i]
			if last_trade_time is not None:
				if rebalance_frequency == RebalanceFrequency.WEEKLY:
					if time.week == last_trade_time.week:
						continue
				elif rebalance_frequency == RebalanceFrequency.MONTHLY:
					if time.month == last_trade_time.month:
						continue
			delta = deltas[i]
			returns = contracts * delta / tick_size * tick_value
			signal = validation_predictions[i]
			long_threshold = np.percentile(signal_history, Configuration.SIGNAL_LONG_PERCENTILE)
			short_threshold = np.percentile(signal_history, Configuration.SIGNAL_SHORT_PERCENTILE)
			if signal > long_threshold and (not Configuration.SIGNAL_SIGN_CHECK or signal > 0):
				# Long trade
				evaluation_results.submit_trade(returns, True)
			elif signal < short_threshold and (not Configuration.SIGNAL_SIGN_CHECK or signal < 0):
				# Short trade
				evaluation_results.submit_trade(returns, False)
			else:
				# No trade
				pass
			signal_history.pop(0)
			signal_history.append(signal)
			assert len(signal_history) == Configuration.SIGNAL_HISTORY
			last_trade_time = time
			actual_returns = y_validation[i]
			signal_returns.append((signal, actual_returns))
		signal_returns = sorted(signal_returns, key=lambda x: x[0])
		sorted_returns = [returns for _signal, returns in signal_returns]
		quintiles = 5
		if len(sorted_returns) == 0:
			# Hack to make stats work
			sorted_returns = quintiles * [0]
		grouped_returns = np.array_split(sorted_returns, quintiles) # type: ignore
		evaluation_results.quantiles = [mean(x) for x in grouped_returns]
		output.append(evaluation_results)

	return output

def filter_data_set(
		x: list[list[float]],
		y: list[float],
		time_values: list[pd.Timestamp],
		time_filter_function: Callable[[pd.Timestamp], bool]
) -> tuple[list[list[float]], list[float]]:
	x_output = []
	y_output = []
	for i in range(len(x)):
		time = time_values[i]
		if time_filter_function(time):
			x_output.append(x[i])
			y_output.append(y[i])
	return x_output, y_output