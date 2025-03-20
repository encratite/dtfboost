import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from statistics import mean
from typing import cast

import pandas as pd

from config import Configuration
from enums import RebalanceFrequency
from results import EvaluationResults
from evaluate import evaluate

def print_hyperparameters(results: list[EvaluationResults]):
	hyperparameters = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	for evaluation_results in results:
		for parameter_name, parameter_value in evaluation_results.parameters.items():
			hyperparameters[evaluation_results.model_name][parameter_name][parameter_value].append(evaluation_results)
	for model_name, parameter_dict in hyperparameters.items():
		print(f"Hyperparameters for {model_name}:")
		for parameter_name, parameter_value_dict in parameter_dict.items():
			print(f"\t{parameter_name}:")
			for parameter_value, evaluation_results in parameter_value_dict.items():
				performance_values = [x.get_annualized_performance() for x in evaluation_results]
				mean_r2_score_training = mean([x.r2_score_training for x in evaluation_results])
				mean_r2_score_validation = mean([x.r2_score_validation for x in evaluation_results])
				mean_absolute_error_training = mean([x.mean_absolute_error_training for x in evaluation_results])
				mean_absolute_error_validation = mean([x.mean_absolute_error_validation for x in evaluation_results])
				mean_performance = mean(performance_values)
				max_performance = max(performance_values)
				performance_string = EvaluationResults.get_performance_string(mean_performance)
				max_performance_string = EvaluationResults.get_performance_string(max_performance)
				print(f"\t\t{parameter_value}: {performance_string} (max {max_performance_string}; R2 scores: {mean_r2_score_training:.3f} training, {mean_r2_score_validation:.3f} validation; MAE: {mean_absolute_error_training:.4f} training, {mean_absolute_error_validation:.4f} validation)")

def print_performance(symbols: list[str], feature_limit: int, results: list[EvaluationResults]):
	total_model_performance = defaultdict(list)
	for evaluation_results in results:
		total_model_performance[evaluation_results.model_name].append(evaluation_results)
	all_model_performance_values = []
	for model_name, evaluation_results in total_model_performance.items():
		all_performance_values = mean([x.get_annualized_performance() for x in evaluation_results])
		mean_r2_score_training = mean([x.r2_score_training for x in evaluation_results])
		mean_r2_score_validation = mean([x.r2_score_validation for x in evaluation_results])
		mean_absolute_error_training = mean([x.mean_absolute_error_training for x in evaluation_results])
		mean_absolute_error_validation = mean([x.mean_absolute_error_validation for x in evaluation_results])
		all_model_performance_values.append(all_performance_values)
		performance_string = EvaluationResults.get_performance_string(all_performance_values)
		print(f"[{model_name}] {performance_string} (R2 scores: {mean_r2_score_training:.3f} training, {mean_r2_score_validation:.3f} validation; MAE: {mean_absolute_error_training:.4f} training, {mean_absolute_error_validation:.4f} validation)")
	mean_model_performance = mean(all_model_performance_values)
	performance_string = EvaluationResults.get_performance_string(mean_model_performance)
	print(f"Mean of all models with a feature limit of {feature_limit}: {performance_string}")
	print(f"Symbols evaluated: {symbols}")

def main() -> None:
	if len(sys.argv) != 7:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbols> <start date> <split date> <end date> <rebalance frequency> <feature limit>")
		print(f"Supported rebalance frequencies: daily, weekly, monthly")
		return
	symbols = [x.strip() for x in sys.argv[1].split(",")]
	start = pd.Timestamp(sys.argv[2])
	split = pd.Timestamp(sys.argv[3])
	end = pd.Timestamp(sys.argv[4])
	rebalance_frequency = cast(RebalanceFrequency, RebalanceFrequency[sys.argv[5].upper()])
	feature_limit = int(sys.argv[6])
	assert start < split < end
	results: list[EvaluationResults]
	if Configuration.ENABLE_MULTIPROCESSING:
		arguments = [(symbol, start, split, end, rebalance_frequency, feature_limit) for symbol in symbols]
		with Pool(cpu_count()) as pool:
			nested_results = pool.starmap(evaluate, arguments)
			results = [item for sublist in nested_results for item in sublist]
	else:
		results = []
		for symbol in symbols:
			results += evaluate(symbol, start, split, end, rebalance_frequency, feature_limit)
	print("")
	print_hyperparameters(results)
	print_performance(symbols, feature_limit, results)

if __name__ == "__main__":
	main()