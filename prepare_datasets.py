import pandas as pd
import numpy as np

dft = pd.read_csv('data/normalised_climchange_adj_temperature_timeseries.csv', index_col=None)
# Final preparation for the dataset
# reshape dataframe to have 64*24 columns
k = 64*24
n = dft.shape[0] // k
data = dft.iloc[:n*k].values.reshape(n, k)
index = dft.index[:-k:k]
dft_reshaped = pd.DataFrame(data=data, index=index)
np.random.seed(42)
# shuffle rows
dft_reshaped = dft_reshaped.sample(frac=1)
dft_reshaped.to_csv('data/final_dataset.csv')

print(dft_reshaped)