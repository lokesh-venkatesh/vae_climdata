import pandas as pd

df = pd.read_csv("data/raw_data.csv", parse_dates=["valid_time"], index_col=None)
print(df.head())
df = df.sort_values("valid_time")
print("")
print("")
print(df.head())