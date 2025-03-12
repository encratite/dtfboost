from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, f1_score as get_f1_score

from enums import FeatureCategory

class TrainingResults:
	precision_values: list[float]
	roc_auc_values: list[float]
	f1_scores: list[float]
	max_precision: float | None
	max_f1_score: float | None
	best_model_parameters: dict[str, int] | None
	best_model: Any | None
	best_model_precision: float | None
	best_model_f1_score: float | None
	best_model_x_training: pd.DataFrame | None
	best_model_y_training: pd.DataFrame | None
	best_model_x_validation: pd.DataFrame | None
	best_model_y_validation: pd.DataFrame | None
	best_feature_categories: frozenset[FeatureCategory] | None
	parameter_f1_scores: defaultdict[str, defaultdict[int, list[float]]]
	feature_category_f1_scores: defaultdict[frozenset[FeatureCategory], list[float]]

	def __init__(self):
		self.precision_values = []
		self.roc_auc_values = []
		self.f1_scores = []
		self.max_precision = None
		self.max_f1_score = None
		self.best_model_parameters = None
		self.best_model = None
		self.best_model_precision = None
		self.best_model_f1_score = None
		self.best_model_x_training = None
		self.best_model_y_training = None
		self.best_model_x_validation = None
		self.best_model_y_validation = None
		self.best_feature_categories = None
		self.parameter_f1_scores = defaultdict(lambda: defaultdict(list))
		self.feature_category_f1_scores = defaultdict(list)

	def submit_model(self, x_training: pd.DataFrame, y_training: pd.DataFrame, x_validation: pd.DataFrame, y_validation: pd.DataFrame, model: Any, model_parameters: dict[str, int], feature_categories: frozenset[FeatureCategory]):
		predictions = model.predict(x_validation)
		predictions = (predictions > 0.5).astype(dtype=np.int8)
		precision = precision_score(y_validation, predictions)
		self.precision_values.append(precision)
		roc_auc = roc_auc_score(y_validation, predictions)
		self.roc_auc_values.append(roc_auc)
		f1_score = get_f1_score(y_validation, predictions)
		self.f1_scores.append(f1_score)

		if self.best_model is None or f1_score > self.best_model_f1_score:
			self.best_model_parameters = model_parameters
			self.best_model = model
			self.best_model_precision = precision
			self.best_model_f1_score = f1_score
			self.best_model_x_training = x_training
			self.best_model_y_training = y_training
			self.best_model_x_validation = x_validation
			self.best_model_y_validation = y_validation
			self.best_feature_categories = feature_categories
		self.max_f1_score = max(f1_score, self.max_f1_score if self.max_f1_score is not None else f1_score)
		self.max_precision = max(precision, self.max_precision if self.max_precision is not None else precision)
		for name, value in model_parameters.items():
			self.parameter_f1_scores[name][value].append(f1_score)
		self.feature_category_f1_scores[feature_categories].append(f1_score)