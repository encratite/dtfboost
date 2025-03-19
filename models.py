from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from config import Configuration

def get_random_forest_models() -> list[Any]:
	n_estimators_values = [
		# 50,
		100,
		# 150,
		200
	]
	criterion_values = [
		"squared_error",
		# "absolute_error",
		# "friedman_mse"
	]
	max_depths_values = [
		None,
		# 3,
		# 4,
		# 5,
		6,
		7,
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

def get_mlp_models():
	hidden_layer_sizes_values = [
		# (16, 8),
		# (16, 8, 4),
		# (20, 10),
		(24, 12),
		# (32, 16),
		# (32, 16, 8),
		# (32, 32),
		# (40, 20),
		# (48, 24),
		# (56, 28),
		# (64, 32),
		# (64, 64)
	]
	activation_values = [
		# "identity",
		"logistic",
		# "tanh",
		# "relu"
	]
	solver_values = [
		"lbfgs",
		# "sgd",
		# "adam"
	]
	max_iter_values = [
		25,
		30,
		40,
		50,
		# 55,
		60,
		# 70,
		# 100,
		200,
		# 500,
		1000,
		# 2000
	]
	models = []
	for hidden_layer_sizes in hidden_layer_sizes_values:
		for activation in activation_values:
			for solver in solver_values:
				for max_iter in max_iter_values:
					parameters = {
						"hidden_layer_sizes": hidden_layer_sizes,
						"activation": activation,
						"solver": solver,
						"max_iter": max_iter
					}
					model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter, random_state=Configuration.SEED)
					models.append(("MLPRegressor", model, parameters, True))
	return models