import sys
from utils import plot_stroke
from models.dummy import load_model, generate_unconditionally, generate_conditionally, recognize_stroke

# model = load_model('unconditional_900', 49)
# model = load_model('unconditional_400', 20)
model = load_model('conditional_400', 10)
print('Loaded model')

# # Unconditional generation
# stroke = generate_unconditionally(model)
# plot_stroke(stroke)

# Conditional generation
stroke = generate_conditionally(model)
plot_stroke(stroke)
