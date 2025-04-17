import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

def adjust_timeseries_for_climchange(dataframe=pd.read_csv('data/raw_data.csv')):
    dataframe['valid_time'] = (pd.to_datetime(dataframe['valid_time'])-pd.to_datetime(dataframe['valid_time'].iloc[0])).dt.total_seconds()/3600
    slope, intercept, r_value, p_value, std_err = linregress(dataframe['valid_time'], dataframe['t2m']) # Perform linear regression

    print(f"The rate of temperature increase per hour is {slope} degrees/hour.")

    most_recent_time = dataframe['valid_time'].max()
    dataframe['t2m_adj'] = dataframe['t2m'] - slope * (dataframe['valid_time'] - most_recent_time)
    return dataframe

'''
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
'''

def normalise_timeseries(dataframe):
    offset = dataframe['t2m_adj'].mean()
    scale = dataframe['t2m_adj'].std()
    dataframe['t2m_norm_adj'] = (dataframe['t2m_adj'] - offset) / scale
    return dataframe

def main():
    df = pd.read_csv('data/raw_data.csv') # Assuming the dataframe has columns 'valid_time' (in hours) and 'Observed'
    df_copy = adjust_timeseries_for_climchange(df)
    df_copy.to_csv('data/adj_data.csv', index=False)
    #df_copy = df_copy.drop(columns=['valid_time'])
    df_desc = df_copy.describe()
    df_desc.to_csv('data/adj_data_stats.csv')
    df_norm = normalise_timeseries(df_copy)
    df_norm.to_csv('data/norm_adj_data.csv', index=False)
    dft_desc = df_norm.describe()
    dft_desc.to_csv('data/norm_adj_data_stats.csv')

if __name__=="__main__":
    main()