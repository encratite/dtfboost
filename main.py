import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from statistics import mean

import pandas as pd

from config import Configuration
from enums import RebalanceFrequency
from evaluate import evaluate
from results import EvaluationResults

def print_newline():
	print("")

def get_additional_stats_strings(evaluation_results: list[EvaluationResults]) -> tuple[str, str]:
	mean_r2_score_training = mean([x.r2_score_training for x in evaluation_results])
	mean_r2_score_validation = mean([x.r2_score_validation for x in evaluation_results])
	mean_absolute_error_training = mean([x.mean_absolute_error_training for x in evaluation_results])
	mean_absolute_error_validation = mean([x.mean_absolute_error_validation for x in evaluation_results])
	r2_score_string = f"R2 scores: training {mean_r2_score_training:.3f}, validation {mean_r2_score_validation:.3f}"
	mean_absolute_error_string = f"MAE: {mean_absolute_error_training:.4f} training, {mean_absolute_error_validation:.4f} validation"
	return r2_score_string, mean_absolute_error_string

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
				mean_performance = mean(performance_values)
				max_performance = max(performance_values)
				performance_string = EvaluationResults.get_performance_string(mean_performance)
				max_performance_string = EvaluationResults.get_performance_string(max_performance)
				r2_score_string, mean_absolute_error_string = get_additional_stats_strings(evaluation_results)
				print(f"\t\t{parameter_value}: {performance_string} (max {max_performance_string}; {r2_score_string}; {mean_absolute_error_string})")

def print_performance(results: list[EvaluationResults], result_category: str | None):
	print_newline()
	if result_category is not None:
		print(f"Results for category \"{result_category}\":")
	total_model_performance = defaultdict(list)
	for evaluation_results in results:
		total_model_performance[evaluation_results.model_type.value].append(evaluation_results)
	all_model_performance_values = []
	buy_and_hold_performance_values = {}
	model_performance_by_asset = defaultdict(list)
	sorted_by_model_type = sorted(total_model_performance.keys())
	for model_type in sorted_by_model_type:
		evaluation_results = total_model_performance[model_type]
		for evaluation_result in evaluation_results:
			buy_and_hold_performance_values[evaluation_result.symbol] = evaluation_result.buy_and_hold_performance
			performance = evaluation_result.get_annualized_performance()
			model_performance_by_asset[evaluation_result.symbol].append(performance)
		all_performance_values = mean([x.get_annualized_performance() for x in evaluation_results])
		quantiles = []
		for i in range(len(evaluation_results[0].quantiles)):
			mean_values = [x.quantiles[i] for x in evaluation_results]
			quantiles.append(mean(mean_values))
		all_model_performance_values.append(all_performance_values)
		mean_model_performance_string = EvaluationResults.get_performance_string(all_performance_values)
		r2_score_string, mean_absolute_error_string = get_additional_stats_strings(evaluation_results)
		quantile_string = EvaluationResults.get_quantiles_string(quantiles)
		print(f"[{evaluation_results[0].model_name}] {mean_model_performance_string} ({r2_score_string}; {mean_absolute_error_string}; quantiles: {quantile_string})")
	buy_and_hold_performance = mean(buy_and_hold_performance_values.values())
	buy_and_hold_performance_string = EvaluationResults.get_performance_string(buy_and_hold_performance)
	mean_model_performance = mean(all_model_performance_values)
	mean_model_performance_string = EvaluationResults.get_performance_string(mean_model_performance)
	print(f"Buy and hold performance: {buy_and_hold_performance_string}")
	print(f"Mean performance of all models: {mean_model_performance_string}")

def print_general_info(
		symbol: str,
		start: pd.Timestamp,
		split: pd.Timestamp,
		end: pd.Timestamp,
		feature_limit: int,
		rebalance_frequency_string: str
) -> None:
	print_newline()
	print(f"Symbol traded: {symbol}")
	print(f"Timestamps: start {get_date_string(start)}, split {get_date_string(split)}, end {get_date_string(end)}")
	if Configuration.USE_PCA:
		print(f"Number of PCA features used: {feature_limit}")
		print(f"Pre-PCA rank filter: {Configuration.PCA_RANK_FILTER}")
	elif Configuration.SELECT_K_BEST:
		print(f"Number of best features selected using \"{Configuration.SELECT_K_BEST_SCORE}\": {feature_limit}")
	else:
		print(f"Number of features used: {feature_limit}")
	print(f"Rebalance frequency: {rebalance_frequency_string}")

def get_date_string(time: pd.Timestamp):
	return time.strftime("%Y-%m-%d")

def main() -> None:
	if len(sys.argv) != 7:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbol> <start date> <split date> <end date> <rebalance frequency> <feature limit>")
		print(f"Supported rebalance frequencies: daily, daily-split, weekly, monthly")
		return
	rebalance_frequency_map = {
		"daily": RebalanceFrequency.DAILY,
		"daily-split": RebalanceFrequency.DAILY_SPLIT,
		"weekly": RebalanceFrequency.WEEKLY,
		"monthly": RebalanceFrequency.MONTHLY,
	}
	symbol = sys.argv[1]
	start = pd.Timestamp(sys.argv[2])
	split = pd.Timestamp(sys.argv[3])
	end = pd.Timestamp(sys.argv[4])
	rebalance_frequency_string = sys.argv[5]
	rebalance_frequency = rebalance_frequency_map[rebalance_frequency_string]
	feature_limit = int(sys.argv[6])
	assert start < split < end
	results: list[EvaluationResults]
	if Configuration.ENABLE_MULTIPROCESSING:
		process_count = cpu_count()
		arguments = [(symbol, start, split, end, rebalance_frequency, feature_limit, process_id, process_count) for process_id in range(process_count)]
		with Pool(process_count) as pool:
			nested_results = pool.starmap(evaluate, arguments)
			results = [item for sublist in nested_results for item in sublist]
	else:
		results = evaluate(symbol, start, split, end, rebalance_frequency, feature_limit, 0, 1)
	print_newline()
	print_hyperparameters(results)
	category_results: defaultdict[int, list[EvaluationResults]] = defaultdict(list)
	for evaluation_results in results:
		category_results[evaluation_results.result_category_id].append(evaluation_results)

	for key, key_evaluation_results in category_results.items():
		result_category = key_evaluation_results[0].result_category
		print_performance(key_evaluation_results, result_category)

	print_general_info(symbol, start, split, end, feature_limit, rebalance_frequency_string)

if __name__ == "__main__":
	main()