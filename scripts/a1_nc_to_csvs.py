import xarray as xr
import pandas as pd
import os

def convert_nc_to_csv(input_nc_file, output_dir_filepath):
    # Load the NetCDF file
    ds = xr.open_dataset(input_nc_file)

    # Convert temperature from Kelvin to Celsius
    ds['t2m'] = ds['t2m'] - 273.15

    # Convert the dataset to a pandas DataFrame
    df = ds.to_dataframe().reset_index()
    # Drop columns 'number' and 'expver' if they exist
    df = df.drop(columns=['number', 'expver'], errors='ignore')

    # Save the DataFrame as a CSV file
    df.to_csv(output_dir_filepath, index=False)

    print(f"Data successfully saved to {output_dir_filepath}")

for i in range(1970, 2019, 3):
    input_nc_file = f'data/t2m_{i}_{i+2}/data_stream-oper_stepType-instant.nc'
    output_dir = f'data/t2m_{i}_{i+2}/t2m_{i}_{i+2}.csv'
    if os.path.exists(output_dir):
        print(f"File {output_dir} already exists. Skipping conversion.")
        continue
    convert_nc_to_csv(input_nc_file, output_dir)

# Merge all the created CSV files into a master CSV file
csv_files = [f'data/t2m_{i}_{i+2}/t2m_{i}_{i+2}.csv' for i in range(1970, 2019, 3)]

# Read and concatenate all CSV files
df_list = [pd.read_csv(csv_file) for csv_file in csv_files if os.path.exists(csv_file)]
master_df = pd.concat(df_list, ignore_index=True)

'''
# Save the merged DataFrame to a new master CSV file
master_csv_path = 'data/master_t2m.csv'
master_df.to_csv(master_csv_path, index=False)
print(f"Master CSV file successfully saved to {master_csv_path}")

df = pd.read_csv("data/master_t2m.csv", parse_dates=["valid_time"])
'''

# Average temperature spatially
df_avg = master_df.groupby("valid_time")["t2m"].mean().reset_index()

# Save the resulting time series to a CSV file
df_avg.to_csv("data/temperature_timeseries.csv", index=False)