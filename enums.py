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