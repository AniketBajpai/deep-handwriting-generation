import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.lstm_multilayer as lstm_multilayer
import models.lstm_singlelayer as lstm_singlelayer
import config
from utils import sentence_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(name, ckpt):
    """ Load model {name} from checkpoint no. ckpt """
    global device
    if name == 'unconditional_900':
        model = lstm_singlelayer.UncondHandwritingGenerator(device, config.INPUT_DIM, config.HIDDEN_DIM_900, config.num_mixture_components)
        ckpt_path = '../checkpoints/unconditional_900/{}.pth'.format(
            ckpt)
    elif name == 'unconditional_900_w_noise':
        model = lstm_singlelayer.UncondHandwritingGenerator(device, config.INPUT_DIM, config.HIDDEN_DIM_900, config.num_mixture_components)
        ckpt_path = '../checkpoints/unconditional_900_w_noise/{}.pth'.format(
            ckpt)
    elif name == 'unconditional_400':
        model = lstm_multilayer.UncondHandwritingGenerator(device, config.INPUT_DIM, config.HIDDEN_DIM_400, config.num_mixture_components)
        ckpt_path = '../checkpoints/unconditional_400/{}.pth'.format(
            ckpt)
    elif name == 'conditional_400':
        model = lstm_multilayer.CondHandwritingGenerator(device, config.INPUT_DIM, config.HIDDEN_DIM_400, config.K, config.num_mixture_components)
        ckpt_path = './checkpoints/conditional_400/{}.pth'.format(
            ckpt)
    elif name == 'conditional_400_w_noise':
        model = lstm_multilayer.CondHandwritingGenerator(device, config.INPUT_DIM, config.HIDDEN_DIM_400, config.K, config.num_mixture_components)
        ckpt_path = '../checkpoints/conditional_400_w_noise/{}.pth'.format(
            ckpt)

    model.eval()
    model.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def sample_y(mdl_parameters):
    r1 = np.random.rand()
    m = config.num_mixture_components
    cumulative_weight = 0
    global device
    y = torch.zeros([1, 3]).to(device)
    e, pi, mu1, mu2, sigma1, sigma2, rho = mdl_parameters

    # Sample from GMM
    for i in range(m):
        cumulative_weight += pi[0, i]
        if cumulative_weight > r1:
            mu1_i = mu1[0, i]
            mu2_i = mu2[0, i]
            sigma1_i = sigma1[0, i]
            sigma2_i = sigma2[0, i]
            rho_i = rho[0, i]
            mvn_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.Tensor([mu1_i, mu2_i]),
                covariance_matrix=torch.Tensor([[sigma1_i**2, rho_i * sigma1_i * sigma2_i], [
                    rho_i * sigma1_i * sigma2_i, sigma2_i**2]])
            )
            y[0, 1:] = mvn_distribution.sample()
            break
    r2 = np.random.rand()
    if e > r2:
        y[0, 0] = 1
    else:
        y[0, 0] = 0
    
    return y


def generate_unconditionally(model, random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)

    np.random.seed(random_seed)
    length_stroke = np.random.randint(400, 1200)
    print('Length of stroke', length_stroke)

    global device
    x = torch.zeros([1, 3]).to(device)  # input to model
    x[0, 0] = 1          # eos_prob at beginning of stroke
    stroke = []            # stores list of parameters for stroke
    stroke.append(x)
    model.init_hidden()

    for t in range(length_stroke):
        mdl_parameters = model(x)
        y = sample_y(mdl_parameters)
        x = y   # Output fed as input at next timestep
        stroke.append(x)    # stroke[t] = x
        
    # Convert stroke to numpy array
    stroke_np = [s.data.cpu().numpy() for s in stroke]
    stroke_np = np.array(stroke_np).squeeze()
    return stroke_np


def generate_conditionally(model, text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    
    np.random.seed(random_seed)
    length_stroke = np.random.randint(400, 1200)
    print('Length of stroke', length_stroke)

    global device
    text_tensor = sentence_to_tensor(text, config.num_chars)
    text_tensor = text_tensor.to(device)
    x = torch.zeros([1, 3]).to(device)  # input to model
    x[0, 0] = 1          # eos_prob at beginning of stroke
    stroke = []            # stores list of parameters for stroke
    attention_maps = []    # stores attention maps at each stroke
    stroke.append(x)
    model.init_hidden()

    for t in range(length_stroke):
        mdl_parameters = model(x, text_tensor)
        y = sample_y(mdl_parameters)
        x = y   # Output fed as input at next timestep
        stroke.append(x)    # stroke[t] = x
        attention_maps.append(model.attention_map)
        
    # Convert stroke to numpy array
    stroke_np = [s.data.cpu().numpy() for s in stroke]
    stroke_np = np.array(stroke_np).squeeze()

    # Convert attention maps to numpy array
    attention_maps_np = [am.data.cpu().numpy() for am in attention_maps]
    attention_maps_np = np.array(attention_maps_np).squeeze()

    return stroke_np, attention_maps_np


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
