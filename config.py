# Data
stroke_train_file_path = './data/strokes_train.npy'
stroke_test_file_path = './data/strokes_test.npy'
text_train_file_path = './data/sentences_train.txt'
text_test_file_path = './data/sentences_test.txt'
num_chars = 76

# Model
# LSTM
INPUT_DIM = 3
HIDDEN_DIM_400 = 400
HIDDEN_DIM_900 = 900
OUTPUT_DIM = 3
lstm_400_dropout = 0.0
lstm_900_dropout = 0.0

# Mixture Model
num_mixture_components = 20

# Attention
K = 10

# Gradient clipping
grad_output_clip = 100
grad_lstm_clip = 10

# Optimizer
lr = 1e-4
alpha = 0.95
momentum = 0.1
epsilon = 1e-4

# Regularization
l2_reg_lambda = 1e-2

# Training
MAX_EPOCHS = 100
examples_per_epoch = 6000
log_freq = 100
plot_freq = 100
num_test_examples = 10
