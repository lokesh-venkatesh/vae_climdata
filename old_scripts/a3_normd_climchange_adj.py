import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

df = pd.read_csv('data/temperature_timeseries.csv') # Assuming the dataframe has columns 'valid_time' (in hours) and 'Observed'
df['valid_time'] = (pd.to_datetime(df['valid_time']) - pd.to_datetime(df['valid_time'].iloc[0])).dt.total_seconds() / 3600
slope, intercept, r_value, p_value, std_err = linregress(df['valid_time'], df['Observed']) # Perform linear regression

print(f"The rate of temperature increase per hour is {slope} degrees/hour.")

most_recent_time = df['valid_time'].max()
df['Climate Adjusted'] = df['Observed'] - slope * (df['valid_time'] - most_recent_time)

df_copy = df.copy()
df_copy.to_csv('data/climchange_adj_temperature_timeseries.csv', index=False)
df_copy = df_copy.drop(columns=['valid_time'])
df_desc = df_copy.describe()
df_desc.to_csv('results/climchange_adj_temperature_timeseries_desc.csv')

# Code for processing and plotting the adjusted temperature:
# Remove entries from 2021
df = df[df['valid_time'] < (2021 - 1970) * 8760]
# Calculate yearly averages
yearly_avg_raw_temp = df.groupby(df['valid_time'] // 8760)['Observed'].mean()
yearly_avg_adj_temp = df.groupby(df['valid_time'] // 8760)['Climate Adjusted'].mean()
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(yearly_avg_raw_temp.index, yearly_avg_raw_temp.values, label='Raw Temperature', color='blue')
plt.plot(yearly_avg_adj_temp.index, yearly_avg_adj_temp.values, label='Adjusted Temperature', color='red')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.xticks(yearly_avg_raw_temp.index, labels=(yearly_avg_raw_temp.index + 1970).astype(int), rotation=90)
plt.title('Yearly Average Temperature Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('results/plot6_impact_of_climate_change.png', dpi=300)

df_mst = df.copy()
offset = round(df_mst['Climate Adjusted'].mean(), 2)
print('offset: ', offset)
scale = round(df_mst['Climate Adjusted'].std(), 2)
print('scale:  ', scale)
dft = (df_mst['Climate Adjusted'] - offset) / scale
dft.to_csv('data/normalised_climchange_adj_temperature_timeseries.csv', index=False)
dft_desc = dft.describe()
dft_desc.to_csv('results/normalised_climchange_adj_temperature_timeseries_desc.csv')

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