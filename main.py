import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from statistics import mean
from typing import Callable

from colorama import init as colorama_init
from colorama import Fore, Style
import pandas as pd
from tabulate import tabulate

from config import Configuration
from enums import RebalanceFrequency
from evaluate import evaluate
from results import EvaluationResults

def print_newline():
	print("")

def get_r2_scores(evaluation_results: list[EvaluationResults]) -> tuple[float, float]:
	mean_r2_score_training = mean([x.r2_score_training for x in evaluation_results])
	mean_r2_score_validation = mean([x.r2_score_validation for x in evaluation_results])
	return mean_r2_score_training, mean_r2_score_validation

def get_mean_absolute_error(evaluation_results: list[EvaluationResults]) -> tuple[float, float]:
	mean_absolute_error_training = mean([x.mean_absolute_error_training for x in evaluation_results])
	mean_absolute_error_validation = mean([x.mean_absolute_error_validation for x in evaluation_results])
	return mean_absolute_error_training, mean_absolute_error_validation

def get_additional_stats_strings(evaluation_results: list[EvaluationResults]) -> tuple[str, str]:
	mean_r2_score_training, mean_r2_score_validation = get_r2_scores(evaluation_results)
	mean_absolute_error_training, mean_absolute_error_validation = get_mean_absolute_error(evaluation_results)
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
				performance_string = EvaluationResults.get_performance_string(mean_performance)
				r2_score_string, mean_absolute_error_string = get_additional_stats_strings(evaluation_results)
				print(f"\t\t{parameter_value}: {performance_string} ({r2_score_string}; {mean_absolute_error_string})")

def format_numeric_value(
	value: float,
	format_string: str,
	positive_threshold: Callable[[float], bool] | None = None,
	negative_threshold: Callable[[float], bool] | None = None
) -> str:
	output = ("{:" + format_string + "}").format(value)
	if positive_threshold is not None and positive_threshold(value):
		output = f"{Fore.GREEN}{output}{Style.RESET_ALL}"
	elif negative_threshold is not None and negative_threshold(value):
		output = f"{Fore.RED}{output}{Style.RESET_ALL}"
	return output

def print_performance(results: list[EvaluationResults], result_category: str | None):
	print_newline()
	if result_category is not None:
		print(f"Results for category \"{result_category}\":")
	total_model_performance = defaultdict(list)
	for evaluation_results in results:
		total_model_performance[evaluation_results.model_type.value].append(evaluation_results)
	all_models_performance_long = []
	all_models_performance_short = []
	all_models_performance_all = []
	buy_and_hold_performance_values = {}
	model_performance_by_asset = defaultdict(list)
	sorted_by_model_type = sorted(total_model_performance.keys())
	table = [[
		"Model",
		"Long",
		"Short",
		"All",
		"SR Long",
		"SR Short",
		"SR All",
		"R2 Train",
		"R2 Val",
		"MAE Train",
		"MAE Val",
		"SigQ1",
		"SigQ2",
		"SigQ3",
		"SigQ4",
		"SigQ5",
		"SigQ5 - SigQ1"
	]]

	def format_mean_absolute_error(mean_absolute_error: float) -> str:
		return format_numeric_value(mean_absolute_error, ".4f", negative_threshold=lambda x: x > 0.25)

	def format_sharpe_ratio(sharpe_ratio: float | None) -> str:
		if sharpe_ratio is not None:
			return format_numeric_value(sharpe_ratio, ".2f", lambda x: x > 0.8, lambda x: x < 0)
		else:
			return "-"

	for model_type in sorted_by_model_type:
		evaluation_results = total_model_performance[model_type]
		for evaluation_result in evaluation_results:
			buy_and_hold_performance_values[evaluation_result.symbol] = (evaluation_result.buy_and_hold_performance, evaluation_result.buy_and_hold_sharpe_ratio)
			performance = evaluation_result.get_annualized_performance()
			model_performance_by_asset[evaluation_result.symbol].append(performance)
		mean_performance_long = mean([x.get_annualized_long_performance() for x in evaluation_results])
		mean_performance_short = mean([x.get_annualized_short_performance() for x in evaluation_results])
		mean_performance_all = mean([x.get_annualized_performance() for x in evaluation_results])
		all_models_performance_long.append(mean_performance_long)
		all_models_performance_short.append(mean_performance_short)
		all_models_performance_all.append(mean_performance_all)

		def get_mean_sharpe_ratio(select: Callable[[EvaluationResults], float]) -> float | None:
			sharpe_ratios = []
			for x in evaluation_results:
				sharpe_ratio = select(x)
				if sharpe_ratio is not None:
					sharpe_ratios.append(sharpe_ratio)
			if len(sharpe_ratios) > 0:
				return mean(sharpe_ratios)
			else:
				return None

		mean_sharpe_ratio_long = get_mean_sharpe_ratio(lambda x: x.get_long_sharpe_ratio())
		mean_sharpe_ratio_short = get_mean_sharpe_ratio(lambda x: x.get_short_sharpe_ratio())
		mean_sharpe_ratio_all = get_mean_sharpe_ratio(lambda x: x.get_total_sharpe_ratio())

		quantiles = []
		for i in range(len(evaluation_results[0].quantiles)):
			mean_values = [x.quantiles[i] for x in evaluation_results]
			quantiles.append(mean(mean_values))
		mean_performance_long_string = EvaluationResults.get_performance_string(mean_performance_long)
		mean_performance_short_string = EvaluationResults.get_performance_string(mean_performance_short)
		mean_performance_all_string = EvaluationResults.get_performance_string(mean_performance_all)
		mean_r2_score_training, mean_r2_score_validation = get_r2_scores(evaluation_results)
		mean_absolute_error_training, mean_absolute_error_validation = get_mean_absolute_error(evaluation_results)
		quantile_cells = EvaluationResults.get_quantile_cells(quantiles)

		table.append([
			evaluation_results[0].model_name,
			mean_performance_long_string,
			mean_performance_short_string,
			mean_performance_all_string,
			format_sharpe_ratio(mean_sharpe_ratio_long),
			format_sharpe_ratio(mean_sharpe_ratio_short),
			format_sharpe_ratio(mean_sharpe_ratio_all),
			f"{mean_r2_score_training:.3f}",
			format_numeric_value(mean_r2_score_validation, ".3f", lambda x: x > 0.01, lambda x: x < 0),
			format_mean_absolute_error(mean_absolute_error_training),
			format_mean_absolute_error(mean_absolute_error_validation)
		] + quantile_cells)
	column_alignment = ("left",) + 16 * ("right",)
	print(tabulate(table, headers="firstrow", tablefmt="simple_outline", disable_numparse=True, colalign=column_alignment))

	def get_mean_string(values: list[float]) -> str:
		mean_value = mean(values)
		return EvaluationResults.get_performance_string(mean_value)

	buy_and_hold_performance = [performance for performance, _sharpe_ratio in buy_and_hold_performance_values.values()][0]
	buy_and_hold_sharpe_ratio = [sharpe_ratio for _performance, sharpe_ratio in buy_and_hold_performance_values.values()][0]

	buy_and_hold_performance_string = EvaluationResults.get_performance_string(buy_and_hold_performance)
	mean_performance_long_string = get_mean_string(all_models_performance_long)
	mean_performance_short_string = get_mean_string(all_models_performance_short)
	mean_performance_all_string = get_mean_string(all_models_performance_all)
	print(f"Buy and hold performance: {buy_and_hold_performance_string}, {format_sharpe_ratio(buy_and_hold_sharpe_ratio)} Sharpe ratio")
	print(f"Mean performance of all models: long {mean_performance_long_string}, short {mean_performance_short_string}, all {mean_performance_all_string}")

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
	if Configuration.ENABLE_MULTIPROCESSING and not Configuration.EDA_MODE:
		process_count = cpu_count()
		arguments = [(
			symbol,
			start,
			split,
			end,
			rebalance_frequency,
			rebalance_frequency_string,
			feature_limit,
			process_id,
			process_count
		) for process_id in range(process_count)]
		with Pool(process_count) as pool:
			nested_results = pool.starmap(evaluate, arguments)
			results = [item for sublist in nested_results for item in sublist]
	else:
		results = evaluate(
			symbol,
			start,
			split,
			end,
			rebalance_frequency,
			rebalance_frequency_string,
			feature_limit,
			0,
			1
		)
	if results is None:
		return

	colorama_init()
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