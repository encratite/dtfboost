from enum import Enum
from typing import Final

class Algorithm(Enum):
	LIGHTGBM: Final[int] = 0
	CATBOOST: Final[int] = 1
	XGBOOST: Final[int] = 3

class PostProcessing(Enum):
	# Apply no post-processing, directly use values from .csv file
	NOMINAL: Final[int] = 0
	# Delta: f(t) - f(t - 1)
	DIFFERENCE: Final[int] = 1
	# Generate two features, the nominal value f(t) and the delta f(t) - f(t - 1)
	NOMINAL_AND_DIFFERENCE: Final[int] = 2
	# Calculate f(t) / f(t - 1) - 1 for the most recent data point
	RATE_OF_CHANGE: Final[int] = 3

class FeatureCategory(Enum):
	SEASONALITY: Final[int] = 0
	TECHNICAL_MOMENTUM: Final[int] = 1
	TECHNICAL_MOVING_AVERAGE: Final[int] = 2
	TECHNICAL_DAYS_SINCE_X: Final[int] = 3
	TECHNICAL_VOLATILITY: Final[int] = 4
	ECONOMIC_INTEREST_RATES: Final[int] = 5
	ECONOMIC_GENERAL: Final[int] = 6
	ECONOMIC_RESOURCES: Final[int] = 7
	ECONOMIC_VOLATILITY: Final[int] = 8
	ECONOMIC_INDEXES: Final[int] = 9
	ECONOMIC_CURRENCIES: Final[int] = 10