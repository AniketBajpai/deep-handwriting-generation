import numpy
from matplotlib import pyplot
import torch
import torch.nn.functional as F


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
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


def compute_mdl_parameters(mdl_parameters_hat):
    e_hat, pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = mdl_parameters_hat

    e = torch.sigmoid(e_hat)
    pi = F.softmax(pi_hat, dim=1)
    mu1 = mu1_hat
    mu2 = mu2_hat
    sigma1 = torch.exp(sigma1_hat)
    sigma2 = torch.exp(sigma2_hat)
    rho = torch.tanh(rho_hat)

    return (e, pi, mu1, mu2, sigma1, sigma2, rho)
