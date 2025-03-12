from itertools import product

import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from config import Configuration
from results import TrainingResults
from enums import FeatureCategory

def train_lightgbm(
		x_training: pd.DataFrame,
		x_validation: pd.DataFrame,
		y_training: pd.DataFrame,
		y_validation: pd.DataFrame,
		optimize: bool,
		feature_categories: frozenset[FeatureCategory] | None,
		results: TrainingResults
) -> None:
	# Hyperparameters
	if optimize:
		num_leaves_values = [20, 30, 40, 50]
		min_data_in_leaf_values = [10, 20, 30]
		max_depth_values = [-1]
		num_iterations_values = [75, 100, 150, 200, 500]
		learning_rate_values = [0.05, 0.1]
	else:
		num_leaves_values = [31]
		min_data_in_leaf_values = [20]
		max_depth_values = [-1]
		num_iterations_values = [100]
		learning_rate_values = [0.1]
	combinations = list(product(num_leaves_values, min_data_in_leaf_values, max_depth_values, num_iterations_values, learning_rate_values))
	for num_leaves, min_data_in_leaf, max_depth, num_iterations, learning_rate in tqdm(combinations, desc="Evaluating hyperparameters", colour="green", disable=not optimize):
		params = {
			"objective": "binary",
			"metric": "binary_logloss",
			"verbosity": -1,
			"num_leaves": num_leaves,
			"min_data_in_leaf": min_data_in_leaf,
			"max_depth": max_depth,
			"num_iterations": num_iterations,
			"learning_rate": learning_rate,
			"seed": Configuration.SEED
		}
		train_dataset = lgb.Dataset(x_training, label=y_training)
		validation_dataset = lgb.Dataset(x_validation, label=y_validation, reference=train_dataset)
		model = lgb.train(params, train_dataset, valid_sets=[validation_dataset])
		model_parameters = {
			"num_leaves": num_leaves,
			"min_data_in_leaf": min_data_in_leaf,
			"max_depth": max_depth,
			"num_iterations": num_iterations,
			"learning_rate": learning_rate,
		}
		results.submit_model(x_training, y_training, x_validation, y_validation, model, model_parameters, feature_categories)

def train_catboost(
		x_training: pd.DataFrame,
		x_validation: pd.DataFrame,
		y_training: pd.DataFrame,
		y_validation: pd.DataFrame,
		optimize: bool,
		feature_categories: frozenset[FeatureCategory] | None,
		results: TrainingResults
) -> None:
	# Hyperparameters
	if optimize:
		iterations_values = [100, 500]
		depth_values = [6, 8, 10]
		learning_rate_values = [0.001, 0.01, 0.1]
	else:
		iterations_values = [100]
		depth_values = [8]
		learning_rate_values = [0.1]
	combinations = list(product(iterations_values, depth_values, learning_rate_values))
	for iterations, depth, learning_rate in tqdm(combinations, desc="Evaluating hyperparameters", colour="green", disabled=not optimize):
		model = CatBoostClassifier(
			iterations=iterations,
			depth=depth,
			learning_rate=learning_rate,
			loss_function="Logloss",
			custom_metric=["AUC"],
			random_seed=Configuration.SEED,
			logging_level="Silent",
			allow_writing_files=False
		)
		model.fit(x_training, y_training, verbose=0)
		model_parameters = {
			"iterations": iterations,
			"depth": depth,
			"learning_rate": learning_rate
		}
		results.submit_model(x_training, y_training, x_validation, y_validation, model, model_parameters, feature_categories)

def train_xgboost(
		x_training: pd.DataFrame,
		x_validation: pd.DataFrame,
		y_training: pd.DataFrame,
		y_validation: pd.DataFrame,
		optimize: bool,
		feature_categories: frozenset[FeatureCategory] | None,
		results: TrainingResults
) -> None:
	# Hyperparameters
	if optimize:
		n_estimators_values = [8, 10, 20, 50, 100, 500]
		max_depth_values = [4, 5, 6, 8, 10]
		eta_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		gamma_values = [0]
		lambda_values = [1]
	else:
		n_estimators_values = [100]
		max_depth_values = [10]
		eta_values = [1.0]
		gamma_values = [0]
		lambda_values = [1]
	combinations = list(product(n_estimators_values, max_depth_values, eta_values, gamma_values, lambda_values))
	for n_estimators, max_depth, eta, gamma, lambda_ in tqdm(combinations, desc="Evaluating hyperparameters", colour="green", disable=not optimize):
		model = XGBClassifier(
			n_estimators=n_estimators,
			max_depth=max_depth,
			eta=eta,
			gamma=gamma,
			lambda_=lambda_,
			objective="binary:logistic",
			verbosity=0,
			seed=Configuration.SEED,
		)
		model.fit(x_training, y_training)
		model_parameters = {
			"n_estimators": n_estimators,
			"max_depth": max_depth,
			"eta": eta,
			"gamma": gamma,
			"lambda": lambda_
		}
		results.submit_model(x_training, y_training, x_validation, y_validation, model, model_parameters, feature_categories)