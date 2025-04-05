import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

df = pd.read_csv('data/temperature_timeseries.csv')
# Assuming the dataframe has columns 'time' (in hours) and 'temperature'
df['valid_time'] = (pd.to_datetime(df['valid_time']) - pd.to_datetime(df['valid_time'].iloc[0])).dt.total_seconds() / 3600

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(df['valid_time'], df['t2m'])

print(f"The rate of temperature increase per hour is {slope} degrees/hour.")

# Adjust the temperature values such that the most recent temperatures are identical
most_recent_time = df['valid_time'].max()
df['adjusted_temperature'] = df['t2m'] - slope * (df['valid_time'] - most_recent_time)

# Save the adjusted DataFrame to a new CSV file
df.to_csv('data/climchange_adj_temperature_timeseries.csv', index=False)

# Remove entries from 2021
df = df[df['valid_time'] < (2021 - 1970) * 8760]

# Calculate yearly averages
yearly_avg_raw_temp = df.groupby(df['valid_time'] // 8760)['t2m'].mean()
yearly_avg_adj_temp = df.groupby(df['valid_time'] // 8760)['adjusted_temperature'].mean()

'''
print("Yearly average raw temperature:")
print(yearly_avg_raw_temp)
print("Yearly average adjusted temperature:")
print(yearly_avg_adj_temp)
'''

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
