from colorama import Fore, Style
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def explore_data(features: dict[str, list[float]], returns: list[float]) -> None:
	print("Enter the name of a feature to analyze:")
	try:
		while True:
			print("> ", end="")
			feature_name = input()
			if feature_name == "":
				return
			if feature_name not in features:
				print(f"{Fore.YELLOW}No such feature{Style.RESET_ALL}")
				continue
			feature = features[feature_name]
			fig = plt.figure(figsize=(12, 10))

			gs = GridSpec(2, 2, height_ratios=[1, 3])

			bins = 50
			feature_hist = fig.add_subplot(gs[0, 0])
			feature_hist.hist(feature, density=True, bins=bins)
			feature_hist.set_title(feature_name)

			returns_title = "Returns"
			returns_hist = fig.add_subplot(gs[0, 1])
			returns_hist.hist(returns, bins=bins)
			returns_hist.set_title(returns_title)

			scatter = fig.add_subplot(gs[1, :])
			scatter.scatter(feature, returns)
			scatter.set_xlabel(feature_name)
			scatter.set_ylabel(returns_title)
			scatter.axhline(y=0, color="0.8", linestyle="solid", linewidth=1)
			scatter.axvline(x=0, color="0.8", linestyle="solid", linewidth=1)

			fig.tight_layout()

			plt.show()
			plt.close()
	except KeyboardInterrupt:
		return