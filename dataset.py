import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import *
from utils import *

time_series_df = pd.read_csv('data/final_timeseries.csv', index_col=None)

def flatten_time_series_into_chunks_of_data(dft):
    # 768 columns = 32 days of hourly data
    k = INPUT_SIZE
    n = dft.shape[0] // k  # number of full 768-length sequences
    dft_reshaped = pd.DataFrame(dft['t2m_norm_adj'].values[:n*k].reshape(n, k), 
                      index=dft['valid_time'].iloc[::k][:n].values, 
                      columns=range(1, k + 1))

    return dft_reshaped

if __name__=="__main__":
    np.random.seed(42)
    dft_reshaped = flatten_time_series_into_chunks_of_data(time_series_df).sample(frac=1)
    dft_reshaped.to_csv(f'data/phoenix_{N_DAYS}days.csv')
    print(dft_reshaped.head())