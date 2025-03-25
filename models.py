from itertools import product
from typing import Any

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, ARDRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from config import Configuration
from wrapper import CatBoostWrapper

def get_linear_models() -> list[tuple[str, Any, dict]]:
	return [
		("LinearRegression", LinearRegression(), {}),
		("LassoCV", LassoCV(max_iter=10000, random_state=Configuration.SEED), {}),
		("ElasticNetCV", ElasticNetCV(max_iter=10000, random_state=Configuration.SEED), {}),
		("ARDRegression", ARDRegression(), {}),
		("BayesianRidge", BayesianRidge(), {}),
	]

def get_random_forest_models() -> list[tuple[str, Any, dict]]:
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
	combinations = product(
		n_estimators_values,
		criterion_values,
		max_depths_values
	)
	for n_estimators, criterion, max_depth in combinations:
		model = RandomForestRegressor(
			n_estimators=n_estimators,
			criterion=criterion,
			max_depth=max_depth,
			random_state=Configuration.SEED
		)
		parameters = {
			"n_estimators": n_estimators,
			"criterion": criterion,
			"max_depth": max_depth
		}
		models.append(("RandomForestRegressor", model, parameters))
	return models

def get_catboost_models() -> list[tuple[str, Any, dict]]:
	iterations_values = [
		10,
		15,
		20,
		25,
		50,
		# 100,
		# 1000,
		# 10000
	]
	depth_values = [
		3,
		4,
		5,
		6,
		# 7,
		# 8,
	]
	learning_rate_values = [
		0.03
	]
	early_stopping_rounds_values = [
		None,
		# 50
	]
	models = []
	combinations = product(
		iterations_values,
		depth_values,
		learning_rate_values,
		early_stopping_rounds_values
	)
	for iterations, depth, learning_rate, early_stopping_rounds in combinations:
		model = CatBoostWrapper(
			iterations=iterations,
			depth=depth,
			learning_rate=learning_rate,
			early_stopping_rounds=early_stopping_rounds,
			random_seed=Configuration.SEED,
			logging_level="Silent",
			allow_writing_files=False
		)
		parameters = {
			"iterations": iterations,
			"depth": depth,
			"learning_rate": learning_rate,
			"early_stopping_rounds": early_stopping_rounds
		}
		models.append(("CatBoostRegressor", model, parameters))
	return models

def get_lightgbm_models() -> list[tuple[str, Any, dict]]:
	num_leaves_values = [
		5,
		10,
		15,
		# 20,
		# 30,
		# 40,
		# 50
	]
	min_data_in_leaf_values = [
		# 5,
		10,
		15,
		20,
		30
	]
	max_depth_values = [
		# -1,
		2,
		3,
		# 4,
		# 5,
		# 6,
		# 7,
	]
	num_iterations_values = [
		20,
		25,
		30,
		40,
		50,
		60,
	]
	learning_rate_values = [
		0.03,
	]
	models = []
	combinations = product(
		num_leaves_values,
		min_data_in_leaf_values,
		max_depth_values,
		num_iterations_values,
		learning_rate_values
	)
	for num_leaves, min_data_in_leaf, max_depth, num_iterations, learning_rate in combinations:
		parameters = {
			"num_leaves": num_leaves,
			"min_data_in_leaf": min_data_in_leaf,
			"max_depth": max_depth,
			"num_iterations": num_iterations,
			"learning_rate": learning_rate,
		}
		model = lgb.LGBMRegressor(
			num_leaves=num_leaves,
			min_data_in_leaf=min_data_in_leaf,
			max_depth=max_depth,
			num_iterations=num_iterations,
			learning_rate=learning_rate,
			verbosity=-1,
			seed=Configuration.SEED
		)
		models.append(("LGBMRegressor", model, parameters))
	return models

def get_xgboost_models() -> list[tuple[str, Any, dict]]:
	n_estimators_values = [
		100,
		200,
		# 1000,
		# 10000
	]
	max_depth_values = [
		2,
		3,
		# 4,
		# 5,
		# 6,
	]
	eta_values = [
		0.1,
		0.15,
		0.2,
		0.25,
		0.3,
		0.35,
		0.4,
		# 0.45,
		# 0.5
	]
	gamma_values = [0]
	lambda_values = [1]
	subsample_values = [
		# 0.1,
		# 0.5,
		# 0.9,
		# 0.95,
		1
	]
	sampling_method_values = [
		"uniform",
		# "gradient_based"
	]
	models = []
	combinations = product(
		n_estimators_values,
		max_depth_values,
		eta_values,
		gamma_values,
		lambda_values,
		subsample_values,
		sampling_method_values
	)
	for n_estimators, max_depth, eta, gamma, lambda_, subsample, sampling_method in combinations:
		parameters = {
			"n_estimators": n_estimators,
			"max_depth": max_depth,
			"eta": eta,
			"gamma": gamma,
			"lambda": lambda_,
			"subsample": subsample,
			"sampling_method": sampling_method,
		}
		model = XGBRegressor(
			n_estimators=n_estimators,
			max_depth=max_depth,
			eta=eta,
			gamma=gamma,
			lambda_=lambda_,
			subsample=subsample,
			sampling_method=sampling_method
		)
		models.append(("XGBRegressor", model, parameters))
	return models

def get_mlp_models() -> list[tuple[str, Any, dict]]:
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
	combinations = product(
		hidden_layer_sizes_values,
		activation_values,
		solver_values,
		max_iter_values,
		learning_rate_init_values
	)
	for hidden_layer_sizes, activation, solver, max_iter, learning_rate_init in combinations:
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
		models.append(("MLPRegressor", model, parameters))
	return models