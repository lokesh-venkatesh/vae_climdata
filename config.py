# number of days the lengths of the data-chunks correspond to
N_DAYS = 64

# dimensions of the input data
INPUT_SIZE = N_DAYS*24
# degree of fourier series for seasonal inputs
DEGREE = 3
# the hours associated with each latent variable
LATENT_SIZE = 4*24

# train and test split
training_ratio = 0.8