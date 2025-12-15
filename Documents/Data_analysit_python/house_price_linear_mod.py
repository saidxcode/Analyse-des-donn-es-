import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing

# Load California housing dataset (built-in, no authentication needed)
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
