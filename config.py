# number of days the lengths of the data-chunks correspond to
N_DAYS = 64

# dimensions of the input data
INPUT_SIZE = N_DAYS*24
# degree of fourier series for seasonal inputs
DEGREE = 3
# the hours associated with each latent variable
LATENT_SIZE = 4*24

# train and test split
TRAINING_RATIO = 0.8

# learning rate, number of epochs and batch size
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 32

# model features
# Parameters
input_shape = None #INPUT_SIZE
latent_dim = None #INPUT_SIZE//LATENT_SIZE
latent_filter = 10
interim_filters = 2*latent_filter