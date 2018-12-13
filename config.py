# Data
stroke_file_path = './data/strokes.npy'
text_file_path = './data/sentences.txt'

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

# Gradient clipping
output_clip = 100
lstm_clip = 10

# Optimizer
lr = 1e-4
alpha = 0.95
momentum = 0.9

# Training
MAX_EPOCHS = 20
log_freq = 100
plot_freq = 100
