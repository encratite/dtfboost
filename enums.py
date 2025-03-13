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
	SEASONALITY: Final[int] = 1

	TECHNICAL_MOMENTUM: Final[int] = 10
	TECHNICAL_VOLUME: Final[int] = 11
	TECHNICAL_OPEN_INTEREST: Final[int] = 12
	TECHNICAL_MOVING_AVERAGE: Final[int] = 13
	TECHNICAL_DAYS_SINCE_X: Final[int] = 14
	TECHNICAL_VOLATILITY: Final[int] = 15
	TECHNICAL_EXPERIMENTAL: Final[int] = 16

	ECONOMIC_INTEREST_RATES: Final[int] = 20
	ECONOMIC_GENERAL: Final[int] = 21
	ECONOMIC_RESOURCES: Final[int] = 22
	ECONOMIC_VOLATILITY: Final[int] = 23
	ECONOMIC_INDEXES: Final[int] = 24
	ECONOMIC_CURRENCIES: Final[int] = 25