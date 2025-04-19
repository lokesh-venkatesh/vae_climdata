# define constants
N_DAYS_PER_CHUNK = 64
INPUT_SIZE = N_DAYS_PER_CHUNK*24 # dimensions of the input data
DEGREE = 3 # degree of fourier series for seasonal inputs
LATENT_SIZE = 4*24 # the hours associated with each latent variable

# Parameters (set these as needed)
#INPUT_SIZE = None  # Replace with actual input size
#LATENT_SIZE = None  # Replace with actual latent size divisor
latent_dim = INPUT_SIZE // LATENT_SIZE
latent_filter = 10
interim_filters = 2 * latent_filter
#DEGREE = None  # Replace with your seasonal degree term

# hyperparams
learning_rate = 0.001
batch_size = 32
epochs = 100
training_ratio = 0.8