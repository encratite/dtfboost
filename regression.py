import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, ARDRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler

from config import Configuration
from enums import RebalanceFrequency
from models import get_random_forest_models, get_mlp_models
from results import EvaluationResults

def perform_regression(
		symbol: str,
		x_training: list[list[float]],
		y_training: list[float],
		x_validation: list[list[float]],
		y_validation: list[float],
		validation_times: list[pd.Timestamp],
		deltas: list[float],
		rebalance_frequency: RebalanceFrequency,
		buy_and_hold_performance: float
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
	slippage = 2 * contracts * (broker_fee + exchange_fee + tick_value)
	models = [
		("LinearRegression", LinearRegression(), {}, False),
		("LassoCV", LassoCV(max_iter=10000, random_state=Configuration.SEED), {}, False),
		("ElasticNetCV", ElasticNetCV(max_iter=10000, random_state=Configuration.SEED), {}, False),
		("ARDRegression", ARDRegression(), {}, False),
		("BayesianRidge", BayesianRidge(), {}, False),
		("RandomForestRegressor", RandomForestRegressor(n_estimators=200, criterion="squared_error", max_depth=6, random_state=Configuration.SEED), {}, False),
		# ("MLPRegressor", MLPRegressor(hidden_layer_sizes=(64, 32), activation="logistic", solver="lbfgs", max_iter=50, random_state=Configuration.SEED), {}, True),
	]

	models += get_random_forest_models()
	models += get_mlp_models()

	scaler = StandardScaler()
	scaler.fit(x_training)

	x_training_scaled = scaler.transform(x_training, copy=True)
	x_validation_scaled = scaler.transform(x_validation, copy=True)

	print(f"[{symbol}] Contracts: {contracts}")
	print(f"[{symbol}] Number of features: {len(x_training[0])}")
	print(f"[{symbol}] Number of samples: {len(x_training)} for training, {len(x_validation)} for validation")
	output = []
	for model_name, model, parameters, enable_scaling in models:
		evaluation_results = EvaluationResults(buy_and_hold_performance, slippage, validation_times[0], validation_times[-1], rebalance_frequency, model_name, parameters)
		if enable_scaling:
			x_training_selected = x_training_scaled
			x_validation_selected = x_validation_scaled
		else:
			x_training_selected = x_training
			x_validation_selected = x_validation
		model.fit(x_training_selected, y_training)
		predictions = model.predict(x_validation_selected)
		last_trade_time: pd.Timestamp | None = None
		for i in range(len(y_validation)):
			time = validation_times[i]
			if last_trade_time is not None:
				if rebalance_frequency == RebalanceFrequency.WEEKLY:
					if time.week == last_trade_time.week:
						continue
				if rebalance_frequency == RebalanceFrequency.MONTHLY:
					if time.month == last_trade_time.month:
						continue
			delta = deltas[i]
			returns = contracts * delta / tick_size * tick_value
			y_predicted = predictions[i]
			long = y_predicted >= 0
			evaluation_results.submit_trade(returns, long)
			last_trade_time = time
		evaluation_results.print_stats(symbol)
		output.append(evaluation_results)

	return output