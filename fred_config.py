from enums import PostProcessing, FeatureCategory, FeatureFrequency

# FRED economic data, this is a complete mess due to the varying offsets
FRED_CONFIG: list[tuple[str, str, PostProcessing, FeatureCategory, FeatureFrequency, str]] = [
	# Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis (DGS10), percentage, daily
	("10-Year T-Note Yield", "DGS10", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_INTEREST_RATES, FeatureFrequency.DAILY, "3:16 PM CDT"),
	# 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity (T10Y3M), percentage, daily
	("10-Year T-Note Minus 3-Month T-Bill Yield", "T10Y3M", PostProcessing.NOMINAL, FeatureCategory.ECONOMIC_INTEREST_RATES, FeatureFrequency.DAILY, "4:02 PM CDT"),
	# Overnight Reverse Repurchase Agreements: Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations (RRPONTSYD), nominal, daily
	("Overnight Reverse Repurchase Agreements", "RRPONTSYD", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_INTEREST_RATES, FeatureFrequency.DAILY, "1:01 PM CDT"),
	# Federal Funds Effective Rate (FEDFUNDS), percentage, monthly
	("Federal Funds Effective Rate", "FEDFUNDS", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_INTEREST_RATES, FeatureFrequency.MONTHLY, "3:17 PM CST"),
	# Federal Funds Effective Rate (DFF), percentage, daily
	("Federal Funds Effective Rate (Daily)", "DFF", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_INTEREST_RATES, FeatureFrequency.DAILY, "3:16 PM CDT"),
	# M2 money supply (M2SL), nominal, monthly
	("M2 Supply", "M2SL", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_INTEREST_RATES, FeatureFrequency.MONTHLY, "12:01 PM CST"),
	# 30-Year Fixed Rate Mortgage Average in the United States (MORTGAGE30US), percentage, weekly
	("30-Year Fixed Rate Mortgage", "MORTGAGE30US", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_INTEREST_RATES, FeatureFrequency.WEEKLY, "11:06 AM CDT"),
	# 15-Year Fixed Rate Mortgage Average in the United States (MORTGAGE15US), percentage, weekly
	("15-Year Fixed Rate Mortgage Average", "MORTGAGE15US", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_INTEREST_RATES, FeatureFrequency.WEEKLY, "11:06 AM CDT"),
	# Real Gross Domestic Product (GDPC1), nominal, quarterly
	("Real Gross Domestic Product", "GDPC1", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.QUARTERLY, "7:58 AM CST"),
	# Personal Consumption Expenditures (PCE), nominal, monthly
	("Personal Consumption Expenditures", "PCE", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:44 AM CST"),
	# Initial unemployment claims (ICSA), nominal value, weekly
	("Initial Unemployment Claims", "ICSA", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.WEEKLY, "7:48 AM CDT"),
	# Unemployment Rate (UNRATE), percentage, monthly
	("Unemployment Rate", "UNRATE", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:46 AM CST"),
	# Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL), nominal, weekly
	("Consumer Price Index", "CPIAUCSL", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.WEEKLY, "7:46 AM CDT"),
	# Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level (WALCL), nominal, weekly
	("Total Assets", "WALCL", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.WEEKLY, "3:37 PM CDT"),
	# All Employees, Total Nonfarm (PAYEMS), nominal, monthly
	("All Employees", "PAYEMS", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:48 AM CST"),
	# Job Openings: Total Nonfarm (JTSJOL), nominal, monthly
	("Job Openings", "JTSJOL", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "9:06 AM CDT"),
	# Producer Price Index by Commodity: Final Demand: Finished Goods Less Foods and Energy (WPSFD4131), nominal, monthly
	("Finished Goods Less Foods and Energy", "WPSFD4131", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:56 AM CDT"),
	# Producer Price Index by Commodity: Final Demand: Finished Goods (WPSFD49207), nominal, monthly
	("Finished Goods", "WPSFD49207", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:57 AM CDT"),
	# Advance Retail Sales: Retail Trade and Food Services (RSAFS), nominal, monthly
	("Retail Trade and Food Services", "RSAFS", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:34 AM CST"),
	# Imports of Goods and Services (IMPGS), nominal, quarterly
	("Imports of Goods and Services", "IMPGS", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.QUARTERLY, "7:58 AM CST"),
	# Exports of Goods and Services (EXPGS), nominal, quarterly
	("Exports of Goods and Services", "EXPGS", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.QUARTERLY, "7:58 AM CST"),
	# Average Weekly Earnings of All Employees, Total Private (CES0500000011), nominal, monthly
	("Average Weekly Earnings", "CES0500000011", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:49 AM CST"),
	# New One Family Houses Sold: United States (HSN1F), nominal, monthly
	("New One Family Houses Sold", "HSN1F", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "9:01 AM CST"),
	# New Privately-Owned Housing Units Started: Total Units (HOUST), nominal, monthly
	("New Privately-Owned Housing Units Started", "HOUST", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:32 AM CST"),
	# Personal Income (PI), nominal, monthly
	("Personal Income", "PI", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:45 AM CST"),
	# Nonfarm Business Sector: Labor Productivity (Output per Hour) for All Workers (PRS85006092), percentage, quarterly
	# Lacks vintage data
	# ("Labor Productivity", "PRS85006092", PostProcessing.NOMINAL, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.QUARTERLY, "7:36 AM CST"),
	# Nonfarm Business Sector: Unit Labor Costs for All Workers (PRS85006112), percentage, quarterly
	# Lacks vintage data
	# ("Unit Labor Costs", "PRS85006112", PostProcessing.NOMINAL, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.QUARTERLY, "7:36 AM CST"),
	# Manufacturers' New Orders: Durable Goods (DGORDER), nominal, monthly
	("Durable Goods New Orders", "DGORDER", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "9:06 AM CST"),
	# Manufacturers' New Orders: Total Manufacturing (AMTMNO), nominal, monthly
	("Total Manufacturing New Orders", "AMTMNO", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "9:06 AM CST"),
	# Industrial Production: Total Index (INDPRO), nominal, monthly
	("Industrial Production - Total Index", "INDPRO", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "8:27 AM CST"),
	# Trade Balance: Goods and Services, Balance of Payments Basis (BOPGSTB), nominal, monthly
	("Trade Balance - Goods and Services", "BOPGSTB", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:51 AM CST"),
	# University of Michigan: Consumer Sentiment (UMCSENT), nominal, monthly
	("University of Michigan Consumer Sentiment", "UMCSENT", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "10:01 AM CST"),
	# Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma (DCOILWTICO), nominal, daily
	("Crude Oil - West Texas Intermediate", "DCOILWTICO", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.DAILY, "12:10 PM CDT"),
	# Crude Oil Prices: Brent - Europe (DCOILBRENTEU), nominal, daily
	("Crude Oil - Brent", "DCOILBRENTEU", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.DAILY, "12:10 PM CDT"),
	# US Regular All Formulations Gas Price (GASREGW), nominal, weekly
	("Gas Price", "GASREGW", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.WEEKLY, "5:05 PM CDT"),
	# Henry Hub Natural Gas Spot Price (DHHNGSP), nominal, daily
	("Henry Hub Natural Gas Spot Price", "DHHNGSP", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.DAILY, "12:36 PM CDT"),
	# Global price of LNG, Asia (PNGASJPUSDM), nominal, monthly
	("Global Price of LNG - Asia", "PNGASJPUSDM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Average Price: Electricity per Kilowatt-Hour in U.S. City Average (APU000072610), nominal, monthly
	# Lacks vintage data
	# ("Price of Electricity", "APU000072610", PostProcessing.DIFFERENCE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.MONTHLY, "7:40 AM CDT"),
	# Global price of Copper (PCOPPUSDM), nominal, monthly
	("Global Price of Copper", "PCOPPUSDM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Global price of Energy index (PNRGINDEXM), nominal, monthly
	("Global Price of Energy Index", "PNRGINDEXM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Global price of Natural gas, EU (PNGASEUUSDM), nominal, monthly
	("Global Price of Natural Gas - EU", "PNGASEUUSDM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Global price of Aluminum (PALUMUSDM), nominal, monthly
	("Global Price of Aluminum", "PALUMUSDM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Global price of Corn (PMAIZMTUSDM), nominal, monthly
	("Global Price of Corn", "PMAIZMTUSDM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Global price of Soybeans (PSOYBUSDM), nominal, monthly
	("Global Price of Soybeans", "PSOYBUSDM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Global price of Food index (PFOODINDEXM), nominal, monthly
	("Global Price of Food index", "PFOODINDEXM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_RESOURCES, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# CBOE Volatility Index: VIX (VIXCLS), nominal, daily
	("CBOE Volatility Index", "VIXCLS", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_VOLATILITY, FeatureFrequency.DAILY, "2:38 PM CDT"),
	# CBOE S&P 500 3-Month Volatility Index (VXVCLS), nominal, daily
	("CBOE S&P 500 3-Month Volatility Index", "VXVCLS", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_VOLATILITY, FeatureFrequency.DAILY, "8:36 AM CDT"),
	# CBOE Gold ETF Volatility Index (GVZCLS), nominal, daily
	("CBOE Gold ETF Volatility Index", "GVZCLS", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_VOLATILITY, FeatureFrequency.DAILY, "8:36 AM CDT"),
	# CBOE Crude Oil ETF Volatility Index (OVXCLS), nominal, daily
	("CBOE Crude Oil ETF Volatility Index", "OVXCLS", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_VOLATILITY, FeatureFrequency.DAILY, "8:36 AM CDT"),
	# CBOE NASDAQ 100 Volatility Index (VXNCLS), nominal, daily
	# ("CBOE NASDAQ 100 Volatility Index", "VXNCLS", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_VOLATILITY, FeatureFrequency.DAILY, "8:36 AM CDT"),
	# CBOE DJIA Volatility Index (VXDCLS), nominal, daily
	# ("CBOE DJIA Volatility Index", "VXDCLS", PostProcessing.NOMINAL_AND_DIFFERENCE, FeatureCategory.ECONOMIC_VOLATILITY, FeatureFrequency.DAILY, "8:36 AM CDT"),
	# Global price of Swine (PPORKUSDM), nominal, monthly
	("Global price of Swine", "PPORKUSDM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Producer Price Index by Commodity: Farm Products: Slaughter Cattle (WPU0131), nominal, monthly
	("Producer Price Index: Slaughter Cattle", "WPU0131", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:56 AM CDT"),
	# Global price of Sugar, No. 11, World (PSUGAISAUSDM), nominal, monthly
	("Global price of Sugar, No. 11, World", "PSUGAISAUSDM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Average Price: Sugar, White, All Sizes (Cost per Pound/453.6 Grams) in U.S. City Average (APU0000715211), nominal, monthly
	# ("Average Price: Sugar, White, All Sizes", "APU0000715211", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:40 AM CDT"),
	# Average Price: Milk, Fresh, Whole, Fortified (Cost per Gallon/3.8 Liters) in U.S. City Average (APU0000709112), nominal, monthly
	# ("Average Price: Milk, Fresh, Whole, Fortified", "APU0000709112", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:40 AM CDT"),
	# Producer Price Index by Commodity: Farm Products: Raw Milk (WPS016), nominal, monthly
	# ("Producer Price Index: Raw Milk", "WPS016", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:57 AM CDT"),
	# Producer Price Index by Industry: Fluid Milk Manufacturing: Fluid Milk and Cream, Bulk Sales (PCU3115113115111), nominal, monthly
	# ("Producer Price Index: Fluid Milk and Cream", "PCU3115113115111", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "8:00 AM CDT"),
	# Producer Price Index by Commodity: Processed Foods and Feeds: Fluid Whole Milk (WPU02310301), nominal, monthly
	# ("Producer Price Index: Fluid Whole Milk", "WPU02310301", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:56 AM CDT"),
	# Producer Price Index by Commodity: Chemicals and Allied Products: Ethanol (Ethyl Alcohol) (WPU06140341), nominal, monthly
	("Producer Price Index by Commodity: Ethanol", "WPU06140341", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:55 AM CDT"),
	# Global price of Cotton (PCOTTINDUSDM), nominal, monthly
	("Global price of Cotton", "PCOTTINDUSDM", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "2:45 PM CDT"),
	# Producer Price Index by Commodity: Special Indexes: Copper and Copper Products (WPUSI019011), nominal, monthly
	("Producer Price Index by Commodit: Copper and Copper Productsy", "WPUSI019011", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:50 AM CDT"),
	# Import Price Index (End Use): Copper (IR14220), nominal, monthly
	("Import Price Index (End Use): Copper", "IR14220", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:37 AM CDT"),
	# Export Price Index (Harmonized System): Copper and Articles Thereof (ID74), nominal, monthly
	("Export Price Index (Harmonized System): Copper and Articles", "ID74", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:37 AM CDT"),
	# Import Price Index (Harmonized System): Copper and Articles Thereof (IP74), nominal, monthly
	("Import Price Index (Harmonized System): Copper and Articles", "IP74", PostProcessing.RATE_OF_CHANGE, FeatureCategory.ECONOMIC_GENERAL, FeatureFrequency.MONTHLY, "7:37 AM CDT"),
]