import warnings
from itertools import product
from statistics import mean
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error as get_mean_absolute_error
from sklearn.metrics import r2_score as get_r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from tqdm import tqdm

from config import Configuration
from data import RegressionDataset, TrainingData
from enums import RebalanceFrequency
from models import get_models
from results import EvaluationResults
from wrapper import RegressionWrapper

def perform_regression(
		symbol: str,
		category_datasets: dict[int, RegressionDataset],
		category_filters: list[tuple[int, str, Callable[[pd.Timestamp], bool]] | None],
		rebalance_frequency: RebalanceFrequency,
		buy_and_hold_performance: float,
		process_id: int,
		process_count: int,
		training_data: TrainingData
	) -> list[EvaluationResults]:
	tick_size, tick_value, contracts, slippage = get_asset_configuration(symbol)
	if Configuration.SHOW_TRADES:
		print("Asset configuration:")
		print(f"\tTick size: {tick_size:.2f}")
		print(f"\tTick value: {tick_value:.2f}")
		print(f"\tContracts: {contracts}")
		print(f"\tSlippage: {slippage:.2f}\n")
	models = get_models()
	transform_data(category_datasets)

	warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
	warnings.filterwarnings("ignore", category=ConvergenceWarning)

	output = []
	tasks = []
	task_id = None
	for model_name, model_type, model, parameters in models:
		if task_id is None:
			task_id = 0
		else:
			task_id += 1
		if task_id % process_count != process_id:
			continue
		tasks.append((model_name, model_type, model, parameters))
	tasks = [a + (b,) for a, b in product(tasks, category_filters)]

	if process_id == 0 and not Configuration.SHOW_TRADES:
		wrapped_tasks = tqdm(tasks, desc="Evaluating models", colour="green")
	else:
		wrapped_tasks = tasks
	for model_name, model_type, model, parameters, category_configuration in wrapped_tasks:
		if category_configuration is None:
			category_id = None
			category_name = None
		else:
			category_id, category_name, _time_filter = category_configuration

		dataset = category_datasets[category_id]
		x_training = dataset.x_training
		y_training = dataset.y_training
		x_validation = dataset.x_validation
		y_validation = dataset.y_validation
		validation_times = dataset.validation_times

		if isinstance(model, RegressionWrapper):
			model.set_validation(x_validation, y_validation)

		model.fit(x_training, y_training)
		training_predictions = model.predict(x_training)
		validation_predictions = model.predict(x_validation)

		r2_score_training = get_r2_score(y_training, training_predictions)
		r2_score_validation = get_r2_score(y_validation, validation_predictions)

		mean_absolute_error_training = get_mean_absolute_error(y_training, training_predictions)
		mean_absolute_error_validation = get_mean_absolute_error(y_validation, validation_predictions)

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
			category_id,
			category_name
		)
		estimate_returns(
			dataset,
			training_predictions,
			validation_predictions,
			rebalance_frequency,
			tick_size,
			tick_value,
			contracts,
			evaluation_results,
			training_data
		)
		output.append(evaluation_results)

	return output

def transform_data(category_datasets: dict[int, RegressionDataset]) -> None:
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
		for dataset in category_datasets.values():
			transformer.fit(dataset.x_training)
			dataset.x_training = transformer.transform(dataset.x_training)
			dataset.x_validation = transformer.transform(dataset.x_validation)

def get_asset_configuration(symbol: str) -> tuple[float, float, int, float]:
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
	return tick_size, tick_value, contracts, slippage

def estimate_returns(
	dataset: RegressionDataset,
	training_predictions: Any,
	validation_predictions: Any,
	rebalance_frequency: RebalanceFrequency,
	tick_size: float,
	tick_value: float,
	contracts: int,
	evaluation_results: EvaluationResults,
	training_data: TrainingData
) -> None:
	y_validation = dataset.y_validation
	validation_times = dataset.validation_times
	deltas = dataset.delta_validation

	treasury_bill = training_data.fred_data["TB3MS"]

	last_trade_time: pd.Timestamp | None = None
	signal_returns: list[tuple[float, float]] = []
	signal_history = training_predictions.tolist()[-Configuration.SIGNAL_HISTORY:]
	for i in range(len(y_validation)):
		time = validation_times[i]
		if last_trade_time is not None:
			if rebalance_frequency == RebalanceFrequency.WEEKLY:
				if time.week == last_trade_time.week:
					continue
			elif rebalance_frequency == RebalanceFrequency.MONTHLY:
				if time.month == last_trade_time.month:
					continue
		risk_free_rate = treasury_bill.get(time) / 100
		delta = deltas[i]
		absolute_return = contracts * delta / tick_size * tick_value
		signal = validation_predictions[i]
		long_threshold = np.percentile(signal_history, Configuration.SIGNAL_LONG_PERCENTILE)
		short_threshold = np.percentile(signal_history, Configuration.SIGNAL_SHORT_PERCENTILE)
		relative_return = y_validation[i]
		if signal > long_threshold and (not Configuration.SIGNAL_SIGN_CHECK or signal > 0):
			# Long trade
			evaluation_results.submit_trade(True, absolute_return, relative_return, risk_free_rate, time)
		elif signal < short_threshold and (not Configuration.SIGNAL_SIGN_CHECK or signal < 0):
			# Short trade
			evaluation_results.submit_trade(False, absolute_return, relative_return, risk_free_rate, time)
		else:
			# No trade
			pass
		signal_history.pop(0)
		signal_history.append(signal)
		assert len(signal_history) == Configuration.SIGNAL_HISTORY
		last_trade_time = time
		signal_returns.append((signal, relative_return))
	signal_returns = sorted(signal_returns, key=lambda x: x[0])
	sorted_returns = [returns for _signal, returns in signal_returns]
	quintiles = 5
	if len(sorted_returns) == 0:
		# Hack to make stats work
		sorted_returns = quintiles * [0]
	grouped_returns = np.array_split(sorted_returns, quintiles)  # type: ignore
	evaluation_results.quantiles = [mean(x) for x in grouped_returns]