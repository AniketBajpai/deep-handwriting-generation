import numpy as np
from matplotlib import pyplot
import torch
import torch.nn.functional as F

import config


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()


def sentence_to_tensor(sentence):
    """ Converts sentence into tensor of one-hot encoded chars """
    chars = list(sentence)
    unique_chars = """!"#'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""
    char_indices = [unique_chars.find(c) for c in chars]
    y_onehot = np.array(char_indices)
    y_onehot = (np.arange(config.num_chars) ==
                y_onehot[:, None]).astype(np.float32)
    y_onehot = torch.from_numpy(y_onehot)
    return y_onehot


def compute_mdl_parameters(mdl_parameters_hat):
    """ Computes Mixture Density Layer parameters """
    e_hat, pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = mdl_parameters_hat

    e = torch.sigmoid(e_hat)
    pi = F.softmax(pi_hat, dim=1)
    mu1 = mu1_hat
    mu2 = mu2_hat
    sigma1 = torch.exp(sigma1_hat)
    sigma2 = torch.exp(sigma2_hat)
    rho = torch.tanh(rho_hat)

    return (e, pi, mu1, mu2, sigma1, sigma2, rho)
