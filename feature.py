from enums import FeatureCategory

class Feature:
	name: str
	category: FeatureCategory
	value: float

	def __init__(self, name: str, category: FeatureCategory, value: float):
		self.name = name
		self.category = category
		self.value = value