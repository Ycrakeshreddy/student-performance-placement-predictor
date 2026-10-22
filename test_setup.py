import pandas as pd
df = pd.read_csv("data/raw/placement_data.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.isnull().sum())