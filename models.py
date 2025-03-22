from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, ARDRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor

from config import Configuration

def get_linear_models() -> list[tuple[str, Any, dict, bool]]:
	return [
		("LinearRegression", LinearRegression(), {}, False),
		("LassoCV", LassoCV(max_iter=10000, random_state=Configuration.SEED), {}, False),
		("ElasticNetCV", ElasticNetCV(max_iter=10000, random_state=Configuration.SEED), {}, False),
		("ARDRegression", ARDRegression(), {}, False),
		("BayesianRidge", BayesianRidge(), {}, False),
	]

def get_random_forest_models() -> list[tuple[str, Any, dict, bool]]:
	n_estimators_values = [
		# 25,
		# 50,
		# 75,
		# 100,
		125,
		# 150,
		# 200
	]
	criterion_values = [
		"squared_error",
		# "absolute_error",
		# "friedman_mse"
	]
	max_depths_values = [
		# None,
		2,
		3,
		4,
		5,
		# 6,
		# 7,
		# 8
	]
	models = []
	for n_estimators in n_estimators_values:
		for criterion in criterion_values:
			for max_depth in max_depths_values:
				model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=Configuration.SEED)
				parameters = {
					"n_estimators": n_estimators,
					"criterion": criterion,
					"max_depth": max_depth
				}
				models.append(("RandomForestRegressor", model, parameters, False))
	return models

def get_mlp_models() -> list[tuple[str, Any, dict, bool]]:
	hidden_layer_sizes_values = [
		(8, 4),
		(8, 4, 2),
		(10, 5),
		(12, 6),
		# (16, 8),
		# (16, 8, 4),
		# (20, 10),
		(20, 10, 5),
		# (24, 12),
		# (24, 12, 6),
		# (32, 16),
		# (40, 20),
		# (40, 20, 10),
		# (48, 24),
		# (48, 24, 12),
		# (56, 28),
		# (64, 32),
	]
	activation_values = [
		# "identity",
		"logistic",
		# "tanh",
		# "relu"
	]
	solver_values = [
		# "lbfgs",
		# "sgd",
		"adam"
	]
	max_iter_values = [
		100,
	]
	learning_rate_init_values = [
		5e-4,
		# 1e-3,
		# 5e-3,
	]
	models = []
	for hidden_layer_sizes in hidden_layer_sizes_values:
		for activation in activation_values:
			for solver in solver_values:
				for max_iter in max_iter_values:
					for learning_rate_init in learning_rate_init_values:
						parameters = {
							"hidden_layer_sizes": hidden_layer_sizes,
							"activation": activation,
							"solver": solver,
							"max_iter": max_iter,
							"learning_rate_init": learning_rate_init
						}
						model = MLPRegressor(
							hidden_layer_sizes=hidden_layer_sizes,
							activation=activation,
							solver=solver,
							max_iter=max_iter,
							learning_rate_init=learning_rate_init,
							learning_rate="adaptive",
							random_state=Configuration.SEED
						)
						models.append(("MLPRegressor", model, parameters, True))
	return models