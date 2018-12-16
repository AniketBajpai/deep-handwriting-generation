import sys
from utils import plot_stroke, plot_attention_map
from models.dummy import load_model, generate_unconditionally, generate_conditionally, recognize_stroke

# model = load_model('unconditional_900', 49)
# model = load_model('unconditional_400', 20)
model = load_model('conditional_400', 25)
# model = load_model('conditional_400_w_noise', 36)
print('Loaded model')

# # Unconditional generation
# stroke = generate_unconditionally(model)
# plot_stroke(stroke)

# Conditional generation
text = 'welcome to lyrebird'
stroke, attention_map = generate_conditionally(model, text)
# plot_stroke(stroke)
plot_attention_map(stroke, text, attention_map)