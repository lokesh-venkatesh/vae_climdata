import os
import xarray as xr
import pandas as pd

# Define the base directory
base_dir = "data/raw_data"
output_dir = "data"
combined_csv_path = os.path.join(output_dir, "raw_data.csv")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize a list to store all dataframes
all_dataframes = []

def save_nc_as_csv(nc_file_path, csv_file_path):
    # Open the .nc file and convert it to a dataframe
    try:
        ds = xr.open_dataset(nc_file_path)
        df = ds.to_dataframe().reset_index()

        # Save the dataframe as a CSV file in the same location
        df.to_csv(csv_file_path, index=False)
        print(f"Saved CSV: {csv_file_path}")

        # Append the dataframe to the list for combining later
        all_dataframes.append(df)
    except Exception as e:
        print(f"Error processing {nc_file_path}: {e}")

def main():
    # Iterate over all subfolders in the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nc"):
                nc_file_path = os.path.join(root, file)
                csv_file_path = os.path.splitext(nc_file_path)[0] + ".csv"
                save_nc_as_csv(nc_file_path=nc_file_path, csv_file_path=csv_file_path)

    # Combine all dataframes into one large dataframe
    if all_dataframes:
        combined_df = pd.concat(all_dataframes)

        # Sort the combined dataframe by time (assuming a 'time' column exists)
        if 'time' in combined_df.columns:
            combined_df = combined_df.sort_values(by='time')
        if 'number' in combined_df.columns:
            combined_df = combined_df.drop('number', axis=1)
        if 'expver' in combined_df.columns:
            combined_df = combined_df.drop('expver', axis=1)

        # To account for time zone difference with respect to GMT
        combined_df['valid_time'] = pd.to_datetime(combined_df['valid_time']) - pd.Timedelta(hours=7)

        # Convert Kelvin to Celsius:
        combined_df['t2m'] = combined_df['t2m']-273.15

        # Average temperature spatially
        combined_df = combined_df.groupby("valid_time")["t2m"].mean().reset_index()

        # Save the combined dataframe as a CSV file
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV saved at: {combined_csv_path}")

        df_desc = combined_df.drop(columns=['valid_time'])
        df_desc = df_desc.describe() # Get descriptive statistics of the temperature data
        df_desc.to_csv("data/raw_data_stats.csv")
    else:
        print("No .nc files were processed.")

if __name__ == "__main__":
    main()