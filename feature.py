from enums import FeatureCategory

class Feature:
	name: str
	category: FeatureCategory
	value: float

	def __init__(self, name: str, category: FeatureCategory, value: float):
		self.name = name
		self.category = category
		self.value = value

	def __repr__(self) -> str:
		return f"{self.name}: {self.value} [{self.category}]"