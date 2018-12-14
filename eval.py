import sys
from utils import plot_stroke
from models.dummy import load_model, generate_unconditionally, generate_conditionally, recognize_stroke

model = load_model('unconditional_400', 20)
print('Loaded model')

# Unconditional generation
stroke = generate_unconditionally(model)
plot_stroke(stroke)
