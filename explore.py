from typing import Any

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt
import seaborn as sns
from colorama import Fore, Style
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import scale, robust_scale, quantile_transform, power_transform
from scipy.stats import spearmanr, pearsonr
from tabulate import tabulate

def explore_data(symbol: str, features: dict[str, list[float]], returns: list[float]) -> None:
	aggregate_handlers = {
		"aggregate": aggregate_junk_features_all,
		"aggregate-pair": aggregate_junk_features_pair
	}
	print("Enter a command:")
	try:
		while True:
			print("> ", end="")
			command = input()
			if command == "":
				return
			tokens = command.split(" ")
			plus_tokens = command.split("+")
			if tokens[0] in aggregate_handlers and len(tokens) == 2:
				command = tokens[0]
				feature_cut_off_string = tokens[1]
				if not feature_cut_off_string.isdecimal():
					print("Invalid feature cut-off")
					continue
				feature_cut_off = int(feature_cut_off_string)
				if feature_cut_off < 0 or feature_cut_off >= len(features):
					print("Feature cut-off out of bounds")
				feature_name, feature = aggregate_handlers[command](features, returns, feature_cut_off)
				tokens = []
			elif len(plus_tokens) == 2:
				feature_name1 = plus_tokens[0].strip()
				feature_name2 = plus_tokens[1].strip()
				if feature_name1 not in features or feature_name2 not in features:
					print("Invalid feature name")
					continue
				feature_name, feature = get_feature_quantile_sum(feature_name1, feature_name2, features)
				tokens = []
			else:
				tokens = command.split(";")
				tokens = [token.strip() for token in tokens]
				if len(tokens) == 0:
					return
				feature_name = tokens[0]
				if feature_name not in features:
					print(f"{Fore.YELLOW}No such feature{Style.RESET_ALL}")
					continue
				feature = features[feature_name]

			pre_transform_pearson = pearsonr(feature, returns)
			pre_transform_spearman = spearmanr(feature, returns) # type: ignore

			transforms = tokens[1:]
			transformed = len(transforms) > 0
			for transform in transforms:
				transform_input = np.array(feature).reshape(-1, 1)
				match transform:
					case "standard":
						feature = scale(transform_input)
						feature = feature.flatten()
					case "robust":
						feature = robust_scale(transform_input)
						feature = feature.flatten()
					case "quantile":
						feature = quantile_transform(transform_input)
						feature = feature.flatten()
					case "yeo-johnson":
						feature = power_transform(transform_input, "yeo-johnson")
						feature = feature.flatten()
					case "box-cox":
						try:
							feature = power_transform(transform_input, "box-cox")
							feature = feature.flatten()
						except ValueError as e:
							print(e)
					case _:
						print(f"Unknown transform: {transform}")

			if transformed:
				post_transform_pearson = pearsonr(feature, returns)
				post_transform_spearman = spearmanr(feature, returns)  # type: ignore
				table = [
					["Description", "Coefficient", "p-value"],
					["Pre-transform Pearson", pre_transform_pearson.statistic, pre_transform_pearson.pvalue],
					["Pre-transform Spearman", pre_transform_spearman.statistic, pre_transform_spearman.pvalue],
					["Post-transform Pearson", post_transform_pearson.statistic, post_transform_pearson.pvalue],
					["Post-transform Spearman", post_transform_spearman.statistic, post_transform_spearman.pvalue],
				]
				print(tabulate(table, headers="firstrow", tablefmt="simple_outline", floatfmt=(None, ".4f", ".4f")))
				transforms_description = ", ".join(transforms)
				annotation_lines = [
					"Pre-transform:",
					f"- ρ: {pre_transform_pearson.statistic:.4f}",
					f"- p-value: {pre_transform_pearson.pvalue:.4f}",
					"",
					"Post-transform:",
					f"- ρ: {post_transform_pearson.statistic:.4f}",
					f"- p-value: {post_transform_pearson.pvalue:.4f}",
					f"- Transform: {transforms_description}",
				]
				annotation_x = 0.81
			else:
				annotation_lines = [
					f"ρ: {pre_transform_pearson.statistic:.4f}",
					f"p-value: {pre_transform_pearson.pvalue:.4f}",
				]
				annotation_x = 0.85
			annotation = "\n".join(annotation_lines)

			fig = plt.figure(figsize=(12, 12))
			fig.canvas.manager.set_window_title(f"[{symbol}] {feature_name}")

			gs = GridSpec(2, 2, height_ratios=[1, 3])

			feature_key = "feature"
			returns_key = "returns"

			data = pd.DataFrame({
				feature_key: feature,
				returns_key: returns
			})

			bins = 40
			feature_histplot = fig.add_subplot(gs[0, 0])
			sns.histplot(ax=feature_histplot, data=data, x=feature_key, bins=bins, kde=True)
			feature_histplot.set(xlabel=feature_name, ylabel="")

			returns_title = f"{symbol} Returns"
			returns_hist = fig.add_subplot(gs[0, 1])
			sns.histplot(ax=returns_hist, data=data, x=returns_key, bins=bins, kde=True)
			returns_hist.set(xlabel=returns_title, ylabel="")

			regplot = fig.add_subplot(gs[1, :])
			sns.regplot(ax=regplot, data=data, x=feature_key, y=returns_key)
			regplot.set(xlabel=feature_name, ylabel=returns_title)
			regplot.text(
				annotation_x,
				0.98,
				annotation,
				transform=regplot.transAxes,
				fontsize=12,
				ha="left",
				va="top",
				bbox=dict(facecolor="white", alpha=0.5)
			)

			fig.tight_layout()

			plt.show()
			plt.close()
	except KeyboardInterrupt:
		return

def get_reshaped_junk_features(features: dict[str, list[float]], returns: list[float], feature_cut_off: int) -> tuple[npt.NDArray, list[tuple[str, list[float], Any]]]:
	correlated_features = []
	for feature_name, feature in features.items():
		pearson = pearsonr(feature, returns)
		correlated_features.append((feature_name, feature, pearson.statistic))
	ranked_features = sorted(correlated_features, key=lambda x: abs(x[2]))[:-feature_cut_off]
	truncated_features = [junk_feature for _feature_name, junk_feature, _pearson in ranked_features]
	reshaped_features = np.array(truncated_features).T
	return reshaped_features, ranked_features

def aggregate_junk_features_all(features: dict[str, list[float]], returns: list[float], feature_cut_off: int) -> tuple[str, list[float]]:
	reshaped_features, _ = get_reshaped_junk_features(features, returns, feature_cut_off)
	quantile_features = quantile_transform(reshaped_features)
	output = []
	for row in quantile_features:
		junk_sum = row.sum()
		output.append(junk_sum)
	return "Junk Feature", output

def aggregate_junk_features_pair(features: dict[str, list[float]], returns: list[float], feature_cut_off: int) -> tuple[str, list[float]]:
	reshaped_features, ranked_features = get_reshaped_junk_features(features, returns, feature_cut_off)
	quantile_features = quantile_transform(reshaped_features).T
	results = []
	feature_names = [feature_name for feature_name, _feature, _pearson in ranked_features]
	for i, f1 in tqdm(list(enumerate(quantile_features)), desc="Evaluating combinations", colour="green"):
		for j, f2 in enumerate(quantile_features):
			if i >= j:
				continue
			f_sum = (np.array(f1) + np.array(f2)).tolist()
			pearson = pearsonr(f_sum, returns)
			results.append((f_sum, pearson.statistic, feature_names[i], feature_names[j]))
	sorted_results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
	print(f"Best untransformed junk feature: {ranked_features[0][2]:.4f}")
	print("Quantile pairs:")
	for i in range(10):
		_, pearson, feature_name1, feature_name2 = sorted_results[i]
		print(f"{i + 1}. {feature_name1} + {feature_name2}: {pearson:.4f}")
	output, _, feature_name1, feature_name2 = sorted_results[0]
	feature_name = f"{feature_name1} + {feature_name2}"
	return feature_name, output

def get_feature_quantile_sum(feature_name1: str, feature_name2: str, features: dict[str, list[float]]) -> tuple[str, list[float]]:
	features1 = features[feature_name1]
	features2 = features[feature_name2]
	reshaped_features = np.array([features1, features2]).T
	quantile_features = quantile_transform(reshaped_features).T
	output = (quantile_features[0] + quantile_features[1]).flatten()
	feature_name = f"{feature_name1} + {feature_name2}"
	return feature_name, output