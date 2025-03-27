import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from colorama import Fore, Style
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import scale, robust_scale, quantile_transform, power_transform
from scipy.stats import spearmanr, pearsonr
from tabulate import tabulate

def explore_data(symbol: str, features: dict[str, list[float]], returns: list[float]) -> None:
	sns.set()
	print("Enter the name of a feature to visualize:")
	try:
		while True:
			print("> ", end="")
			command = input()
			if command == "":
				return
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