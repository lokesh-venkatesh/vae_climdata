import xarray as xr
import pandas as pd
import os

def convert_nc_to_csv(input_nc_file, output_dir_filepath):
    ds = xr.open_dataset(input_nc_file) # Load the NetCDF file
    ds['t2m'] = ds['t2m'] - 273.15 # Convert temperature from Kelvin to Celsius
    df = ds.to_dataframe().reset_index() # Convert the dataset to a pandas DataFrame
    df = df.drop(columns=['number', 'expver'], errors='ignore') # Drop columns 'number' and 'expver' if they exist
    df.to_csv(output_dir_filepath, index=False) # Save the DataFrame as a CSV file
    print(f"Data successfully saved to {output_dir_filepath}")

for i in range(1970, 2019, 3):
    input_nc_file = f'data/raw_data/t2m_{i}_{i+2}/data_stream-oper_stepType-instant.nc'
    output_dir = f'data/raw_data/t2m_{i}_{i+2}/t2m_{i}_{i+2}.csv'
    if os.path.exists(output_dir):
        print(f"File {output_dir} already exists. Skipping conversion.")
        continue
    convert_nc_to_csv(input_nc_file, output_dir)

csv_files = [f'data/raw_data/t2m_{i}_{i+2}/t2m_{i}_{i+2}.csv' for i in range(1970, 2019, 3)] # Merge all the created CSV files into a master CSV file

df_list = [pd.read_csv(csv_file) for csv_file in csv_files if os.path.exists(csv_file)] # Read and concatenate all CSV files
master_df = pd.concat(df_list, ignore_index=True)

df_mst = master_df.copy()
df_mst['valid_time'] = pd.to_datetime(df_mst['valid_time']) - pd.Timedelta(hours=7)

df_avg = df_mst.groupby("valid_time")["t2m"].mean().reset_index() # Average temperature spatially
df_avg.rename(columns={"t2m": "Observed"}, inplace=True)

df_avg.to_csv("data/temperature_timeseries.csv", index=False) # Save the resulting time series to a CSV file

df_desc = df_avg.drop(columns=['valid_time'])
df_desc = df_desc.describe() # Get descriptive statistics of the temperature data
df_desc.to_csv("results/temperature_timeseries_stats.csv")