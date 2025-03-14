from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
		# iterations_values = [1000, 2000, 5000]
		iterations_values = [100]
		depth_values = [3]
		learning_rate_values = [0.02]
	else:
		iterations_values = [100]
		depth_values = [7]
		learning_rate_values = [0.1]
	combinations = list(product(iterations_values, depth_values, learning_rate_values))
	for iterations, depth, learning_rate in tqdm(combinations, desc="Evaluating hyperparameters", colour="green", disable=True):
		model = CatBoostClassifier(
			iterations=iterations,
			depth=depth,
			learning_rate=learning_rate,
			loss_function="Logloss",
			custom_metric=["AUC"],
			random_seed=Configuration.SEED,
			logging_level="Silent",
			l2_leaf_reg=8,
			subsample=0.7,
			bootstrap_type="Bernoulli",
			allow_writing_files=False
		)
		training_pool = Pool(data=x_training, label=y_training)
		validation_pool = Pool(data=x_validation, label=y_validation)
		model.fit(training_pool, eval_set=validation_pool, verbose=0)
		model_parameters = {
			"iterations": iterations,
			"depth": depth,
			"learning_rate": learning_rate
		}
		results.submit_model(x_training, y_training, x_validation, y_validation, model, model_parameters, feature_categories)
		show_training_progress(model)

def show_training_progress(model) -> None:
	evals_result = model.get_evals_result()
	train_loss = evals_result["learn"]["Logloss"]
	test_loss = evals_result["validation"]["Logloss"]

	iterations = np.arange(1, len(train_loss) + 1)

	plt.figure(figsize=(8, 5))
	plt.plot(iterations, train_loss, label="Training Loss", color="blue")
	plt.plot(iterations, test_loss, label="Validation Loss", color="red")
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.title("CatBoost")
	plt.legend()
	plt.grid(True)
	plt.show()

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
		# n_estimators_values = [7, 8, 9, 10]
		n_estimators_values = [
			100
			# 1000,
			# 10000
		]
		max_depth_values = [6, 8, 10]
		eta_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
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
	else:
		n_estimators_values = [100]
		max_depth_values = [8]
		eta_values = [0.35]
		gamma_values = [0]
		lambda_values = [1]
		subsample_values = [1]
		sampling_method_values = ["uniform"]
	combinations = list(product(n_estimators_values, max_depth_values, eta_values, gamma_values, lambda_values, subsample_values, sampling_method_values))
	for n_estimators, max_depth, eta, gamma, lambda_, subsample, sampling_method in tqdm(combinations, desc="Evaluating hyperparameters", colour="green", disable=not optimize):
		model = XGBClassifier(
			n_estimators=n_estimators,
			max_depth=max_depth,
			eta=eta,
			gamma=gamma,
			lambda_=lambda_,
			subsample=subsample,
			sampling_method=sampling_method,
			objective="binary:logistic",
			verbosity=0,
			seed=Configuration.SEED,
			device="cuda" if sampling_method == "gradient_based" else "cpu"
		)
		model.fit(x_training, y_training)
		model_parameters = {
			"n_estimators": n_estimators,
			"max_depth": max_depth,
			"eta": eta,
			"gamma": gamma,
			"lambda": lambda_,
			"subsample": subsample,
			"sampling_method": sampling_method,
		}
		results.submit_model(x_training, y_training, x_validation, y_validation, model, model_parameters, feature_categories)