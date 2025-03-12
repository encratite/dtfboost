import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

from config import Configuration
from enums import FeatureCategory
from results import TrainingResults

def generate_stats(
		symbol: str,
		results: TrainingResults,
		hyperparameters: bool = False,
		feature_categories: bool = False
):
	label_distribution = get_label_distribution(results.best_model_y_validation)
	print(f"Number of samples in training data: {results.best_model_x_training.shape[0]}")
	print(f"Number of samples in validation data: {results.best_model_x_validation.shape[0]}")
	print(f"Number of features: {results.best_model_x_training.shape[1]}")
	print(f"Positive labels: {label_distribution[1]:.1%}")
	print(f"Negative labels: {label_distribution[0]:.1%}")

	mean_precision = mean(results.precision_values)
	mean_roc_auc = mean(results.roc_auc_values)
	mean_f1 = mean(results.f1_scores)
	print(f"Mean precision: {mean_precision:.1%}")
	print(f"Mean ROC-AUC: {mean_roc_auc:.3f}")
	print(f"Mean F1 score: {mean_f1:.3f}")
	print(f"Maximum precision: {results.max_precision:.1%}")
	print(f"Maximum F1 score: {results.max_f1_score:.3f}")

	if hyperparameters:
		print("Mean F1 scores of hyperparameters:")
		for name, values in results.parameter_f1_scores.items():
			print(f"\t{name}:")
			for value, f1_values in values.items():
				print(f"\t\t{value}: {mean(f1_values):.3f}")

		print(f"Best model precision: {results.best_model_precision:.1%}")
		print(f"Best model F1 score: {results.best_model_f1_score:.3f}")
		print("Best hyperparameters:")
		print(results.best_model_parameters)

	if feature_categories:
		enum_f1_scores = []
		for enum in FeatureCategory:
			f1_scores = []
			for feature_categories, f1_score in results.feature_category_f1_scores.items():
				if enum in feature_categories:
					f1_scores += f1_score
			if len(f1_scores) > 0:
				mean_f1_score = mean(f1_scores)
				enum_f1_scores.append((enum, mean_f1_score))
		enum_f1_scores = sorted(enum_f1_scores, key=lambda x: x[1], reverse=True)
		if len(enum_f1_scores) > 0:
			print("Mean F1 scores of feature categories:")
			i = 1
			for enum, mean_f1_score in enum_f1_scores:
				print(f"\t{i}. {enum.name}: {mean_f1_score:.3f}")
				i += 1
		print("Best combination of feature categories:")
		for enum in results.best_feature_categories:
			print(f"\t{enum.name}")

	# Render SHAP summary
	explainer = shap.TreeExplainer(results.best_model)
	x_all = pd.concat([results.best_model_x_training, results.best_model_x_validation], ignore_index=True)
	shap_values = explainer(x_all)
	shap.summary_plot(shap_values, x_all, max_display=30, show=False, plot_size=(12, 12))
	save_plot(symbol, "SHAP Summary")

	# Mean feature importance values
	df = pd.DataFrame({
		"Feature": x_all.columns,
		"Mean Absolute SHAP": np.mean(np.abs(shap_values.values), axis=0)
	})
	df = df.sort_values(by="Mean Absolute SHAP", ascending=False)
	csv_path = os.path.join(Configuration.PLOT_DIRECTORY, symbol, f"Feature Importance.csv")
	df.to_csv(csv_path, index=False)

	# SHAP dependence plots
	if Configuration.GENERATE_DEPENDENCE_PLOTS:
		# Workaround for too many PyPlot figures being created for some reason
		# The plt.close() in save_plot doesn't seem to do the trick
		plt.rcParams["figure.max_open_warning"] = 1000
		for feature in tqdm(range(x_all.shape[1]), desc="Generating dependence plots", colour="green"):
			plt.figure(figsize=(14, 8))
			shap.dependence_plot(feature, shap_values.values, x_all, show=False)
			save_plot(symbol, f"Dependence", x_all.columns[feature])

def save_plot(*tokens: str) -> None:
	directories = tokens[:-1]
	name = tokens[-1]
	directory = os.path.join(Configuration.PLOT_DIRECTORY, *directories)
	path = Path(directory)
	path.mkdir(parents=True, exist_ok=True)
	plot_path = os.path.join(directory, f"{name}.png")
	plt.savefig(plot_path)
	plt.close()

def get_label_distribution(y_validation: pd.DataFrame) -> defaultdict[int, float]:
	output = defaultdict(float)
	for label in y_validation.iloc[:, 0]:
		output[label] += 1.0
	for label in output:
		output[label] /= y_validation.shape[0]
	return output