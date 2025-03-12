import pandas as pd

from data import TrainingData
from enums import PostProcessing
from technical import get_rate_of_change

def get_fred_features(yesterday: pd.Timestamp, data: TrainingData) -> tuple[list[str], list[float]]:
	# FRED economic data
	fred_config = [
		# Initial unemployment claims (ICSA), nominal value, weekly
		("Initial Unemployment Claims", "ICSA", PostProcessing.RATE_OF_CHANGE),
		# Unemployment Rate (UNRATE), percentage, monthly
		("Unemployment Rate", "UNRATE", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis (DGS10), percentage, daily
		("10-Year T-Note Yield", "DGS10", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity (T10Y3M), percentage, daily
		("10-Year T-Note Minus 3-Month T-Bill Yield", "T10Y3M", PostProcessing.NOMINAL),
		# 30-Year Fixed Rate Mortgage Average in the United States (MORTGAGE30US), percentage, weekly
		("30-Year Fixed Rate Mortgage", "MORTGAGE30US", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# 15-Year Fixed Rate Mortgage Average in the United States (MORTGAGE15US), percentage, weekly
		("15-Year Fixed Rate Mortgage Average", "MORTGAGE15US", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Overnight Reverse Repurchase Agreements: Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations (RRPONTSYD), nominal, daily
		("Overnight Reverse Repurchase Agreements", "RRPONTSYD", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Federal Funds Effective Rate (FEDFUNDS), percentage, monthly
		("Federal Funds Effective Rate", "FEDFUNDS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Federal Funds Effective Rate (DFF)
		("Federal Funds Effective Rate (Daily)", "DFF", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# M2 money supply (M2SL), nominal, monthly
		("M2 Supply", "M2SL", PostProcessing.RATE_OF_CHANGE),
		# Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL), nominal, weekly
		("Consumer Price Index", "CPIAUCSL", PostProcessing.RATE_OF_CHANGE),
		# Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level (WALCL), nominal, weekly
		("Total Assets", "WALCL", PostProcessing.RATE_OF_CHANGE),
		# Real Gross Domestic Product (GDPC1), nominal, quarterly
		("Real Gross Domestic Product", "GDPC1", PostProcessing.RATE_OF_CHANGE),
		# All Employees, Total Nonfarm (PAYEMS), nominal, monthly
		("All Employees", "PAYEMS", PostProcessing.RATE_OF_CHANGE),
		# Job Openings: Total Nonfarm (JTSJOL), nominal, monthly
		("Job Openings", "JTSJOL", PostProcessing.RATE_OF_CHANGE),
		# Producer Price Index by Commodity: Final Demand: Finished Goods Less Foods and Energy (WPSFD4131), nominal, monthly
		("Finished Goods Less Foods and Energy", "WPSFD4131", PostProcessing.RATE_OF_CHANGE),
		# Producer Price Index by Commodity: Final Demand: Finished Goods (WPSFD49207), nominal, monthly
		("Finished Goods", "WPSFD49207", PostProcessing.RATE_OF_CHANGE),
		# Advance Retail Sales: Retail Trade and Food Services (RSAFS), nominal, monthly
		("Retail Trade and Food Services", "RSAFS", PostProcessing.RATE_OF_CHANGE),
		# Imports of Goods and Services (IMPGS), nominal, quarterly
		("Imports of Goods and Services", "IMPGS", PostProcessing.RATE_OF_CHANGE),
		# Exports of Goods and Services (EXPGS), nominal, quarterly
		("Exports of Goods and Services", "EXPGS", PostProcessing.RATE_OF_CHANGE),
		# Average Weekly Earnings of All Employees, Total Private (CES0500000011), nominal, monthly
		("Average Weekly Earnings", "CES0500000011", PostProcessing.RATE_OF_CHANGE),
		# Nonfarm Business Sector: Labor Productivity (Output per Hour) for All Workers (PRS85006092), percentage, quarterly
		("Labor Productivity", "PRS85006092", PostProcessing.NOMINAL),
		# Nonfarm Business Sector: Unit Labor Costs for All Workers (PRS85006112), percentage, quarterly
		("Unit Labor Costs", "PRS85006112", PostProcessing.NOMINAL),
		# Manufacturers' New Orders: Durable Goods (DGORDER), nominal, monthly
		("Durable Goods New Orders", "DGORDER", PostProcessing.RATE_OF_CHANGE),
		# Manufacturers' New Orders: Total Manufacturing (AMTMNO), nominal, monthly
		("Total Manufacturing New Orders", "AMTMNO", PostProcessing.RATE_OF_CHANGE),
		# New One Family Houses Sold: United States (HSN1F), nominal, monthly
		("New One Family Houses Sold", "HSN1F", PostProcessing.RATE_OF_CHANGE),
		# New Privately-Owned Housing Units Started: Total Units (HOUST), nominal, monthly
		("New Privately-Owned Housing Units Started", "HOUST", PostProcessing.RATE_OF_CHANGE),
		# Industrial Production: Total Index (INDPRO), nominal, monthly
		("Industrial Production - Total Index", "INDPRO", PostProcessing.RATE_OF_CHANGE),
		# Personal Income (PI), nominal, monthly
		("Personal Income", "PI", PostProcessing.RATE_OF_CHANGE),
		# Personal Consumption Expenditures (PCE), nominal, monthly
		("Personal Consumption Expenditures", "PCE", PostProcessing.RATE_OF_CHANGE),
		# Trade Balance: Goods and Services, Balance of Payments Basis (BOPGSTB), nominal, monthly
		("Trade Balance - Goods and Services", "BOPGSTB", PostProcessing.RATE_OF_CHANGE),
		# University of Michigan: Consumer Sentiment (UMCSENT), nominal, monthly
		("University of Michigan Consumer Sentiment", "UMCSENT", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma (DCOILWTICO), nominal, daily
		("Crude Oil - West Texas Intermediate", "DCOILWTICO", PostProcessing.RATE_OF_CHANGE),
		# Crude Oil Prices: Brent - Europe (DCOILBRENTEU), nominal, daily
		("Crude Oil - Brent", "DCOILBRENTEU", PostProcessing.RATE_OF_CHANGE),
		# US Regular All Formulations Gas Price (GASREGW), nominal, weekly
		("Gas Price", "GASREGW", PostProcessing.RATE_OF_CHANGE),
		# Henry Hub Natural Gas Spot Price (DHHNGSP), nominal, daily
		("Henry Hub Natural Gas Spot Price", "DHHNGSP", PostProcessing.RATE_OF_CHANGE),
		# Global price of LNG, Asia (PNGASJPUSDM), nominal, monthly
		("Global Price of LNG - Asia", "PNGASJPUSDM", PostProcessing.RATE_OF_CHANGE),
		# Average Price: Electricity per Kilowatt-Hour in U.S. City Average (APU000072610), nominal, monthly
		("Price of Electricity", "APU000072610", PostProcessing.DIFFERENCE),
		# Global price of Copper (PCOPPUSDM), nominal, monthly
		("Global Price of Copper", "PCOPPUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Energy index (PNRGINDEXM), nominal, monthly
		("Global Price of Energy Index", "PNRGINDEXM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Natural gas, EU (PNGASEUUSDM), nominal, monthly
		("Global Price of Natural Gas - EU", "PNGASEUUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Aluminum (PALUMUSDM), nominal, monthly
		("Global Price of Aluminum", "PALUMUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Corn (PMAIZMTUSDM), nominal, monthly
		("Global Price of Corn", "PMAIZMTUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Soybeans (PSOYBUSDM), nominal, monthly
		("Global Price of Soybeans", "PSOYBUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Food index (PFOODINDEXM), nominal, monthly
		("Global Price of Food index", "PFOODINDEXM", PostProcessing.RATE_OF_CHANGE),
		# CBOE Volatility Index: VIX (VIXCLS), nominal, daily
		("CBOE Volatility Index", "VIXCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE S&P 500 3-Month Volatility Index (VXVCLS), nominal, daily
		("CBOE S&P 500 3-Month Volatility Index", "VXVCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE Gold ETF Volatility Index (GVZCLS), nominal, daily
		("CBOE Gold ETF Volatility Index", "GVZCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE Crude Oil ETF Volatility Index (OVXCLS), nominal, daily
		("CBOE Crude Oil ETF Volatility Index", "OVXCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE NASDAQ 100 Volatility Index (VXNCLS), nominal, daily
		("CBOE NASDAQ 100 Volatility Index", "VXNCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE DJIA Volatility Index (VXDCLS), nominal, daily
		("CBOE DJIA Volatility Index", "VXDCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
	]
	feature_names = []
	features = []
	for config in fred_config:
		name, symbol, post_processing = config
		match post_processing:
			case PostProcessing.NOMINAL:
				value = data.fred_data[symbol].get(yesterday)
				features.append(value)
				feature_names.append(name)
			case PostProcessing.DIFFERENCE:
				values = data.fred_data[symbol].get(yesterday, count=2)
				difference = values[0] - values[1]
				features.append(difference)
				feature_names.append(f"{name} (Delta)")
			case PostProcessing.NOMINAL_AND_DIFFERENCE:
				values = data.fred_data[symbol].get(yesterday, count=2)
				nominal_value = values[0]
				difference = values[0] - values[1]
				features += [nominal_value, difference]
				feature_names += [
					name,
					f"{name} (Delta)"
				]
			case PostProcessing.RATE_OF_CHANGE:
				values = data.fred_data[symbol].get(yesterday, count=2)
				rate = get_rate_of_change(values[0], values[1])
				features.append(rate)
				feature_names.append(f"{name} (Rate of Change)")
	return feature_names, features