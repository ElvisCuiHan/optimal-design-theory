import pyswarms as ps
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import argparse
import json

def logit(x, alpha, beta):
    eta = alpha + beta * x
    return np.exp(eta) / (1 + np.exp(eta)) ** 2
def probit(x, alpha, beta):
    eta = alpha + beta * x
    Phi = stats.norm.cdf(eta)
    return np.exp(-eta ** 2) / (2 * np.pi * Phi * (1 - Phi))
def cox(x, alpha, beta):
    eta = alpha + beta * x
    return (np.exp(2 * eta - np.exp(eta))) / (1 - np.exp(-np.exp(eta)))
def laplace(x, alpha, beta):
    eta = alpha + beta * x
    return np.exp(-2 * np.abs(eta) - np.log(stats.laplace.cdf(eta)) - np.log(stats.laplace.cdf(-eta)))
def student(x, alpha, beta):
    eta = alpha + beta * x
    return stats.t.pdf(eta, 2) ** 2 / (stats.t.cdf(eta, 2) * stats.t.pdf(-eta, 2))

def info(x, alpha, beta, link="logit"):
    hi = np.zeros((2, 2))
    if link == "logit":
        weight = logit(x, alpha, beta)
    elif link == "probit":
        weight = probit(x, alpha, beta)
    elif link == "cox":
        weight = cox(x, alpha, beta)
    elif link == "laplace":
        weight = laplace(x, alpha, beta)
    elif link == "t":
        weight = student(x, alpha, beta)
    else:
        print("Please write your own code for your own link function!")
        return

    hi[0, 0] = np.sum(weight)
    hi[0, 1] = hi[1, 0] = np.sum(weight * x)
    hi[1, 1] = np.sum(weight * (x ** 2))
    return hi
def D_optim(b, **kwargs):
    """D-optim design

    Parameters
    ----------
    b : numpy.ndarray
        sets of inputs shape :code:'(n_particles, dimensions)'
        usually for a simple logistic model, dimension is 8.

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:'(n_particles, )'
    """
    alpha, beta, link = kwargs.values()

    n, d = b.shape
    loss = np.zeros(n)

    for i in range(n):
        m = np.zeros((2, 2))
        x = b[i, :(d // 2)]
        p = b[i, (d // 2):]
        # p[-1] = 1 - np.sum(p) + p[-1]
        p = p / np.sum(p)

        for j in range((d // 2)):
            # print(info(x[j], alpha, beta))
            m += p[j] * info(x[j], alpha, beta, link)  # p[j] * (ca.dot(ca.T))#

        # m = np.linalg.inv(m)

        loss[i] = np.linalg.det(m)

    return -loss
def sensitivity(x, design, alpha, beta, link="logit"):
    """
    This function calculates the sensitivity function of a design.
    """
    n = x.shape[0]
    d = len(design)
    design_point = design[:(d // 2)]
    p = design[(d // 2):]
    M = np.zeros((2, 2))
    for j in range((d // 2)):
        M += p[j] * info(design_point[j], alpha, beta, link)
    inv_M = np.linalg.inv(M)
    output = np.zeros(n)

    for i in range(n):
        if link == "logit":
            ca = logit(x[i, 1], alpha, beta)
        elif link == "probit":
            ca = probit(x[i, 1], alpha, beta)
        elif link == "cox":
            ca = cox(x[i, 1], alpha, beta)
        elif link == "laplace":
            ca = laplace(x[i, 1], alpha, beta)
        elif link == "t":
            ca = student(x, alpha, beta)
        output[i] = (ca * x[i, :]).dot(inv_M).dot(x[i, :]) - 2
    return output

def optim_design(file_path):
    pars = json.load(open(file_path))
    n = pars["n_particles"]  # number of particles
    d = pars["n_design_points"] * 2  # dimension of the problem
    n_iter = pars["n_iter"]
    b = np.random.random((n, d))

    link_type = pars["link_type"]

    alpha = pars["a"]
    beta = pars["b"]

    bounds = [tuple(np.concatenate([[-10.] * (d // 2), [0] * (d // 2)])),
              tuple(np.concatenate([[10] * (d // 2), [1] * (d // 2)]))]

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=n, dimensions=d, options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(D_optim, iters=n_iter, alpha=alpha, beta=beta, link=link_type)
    best_pos[(d // 2):] = best_pos[(d // 2):] / np.sum(best_pos[(d // 2):])

    num = 1000
    low = pars["low_bound"]
    upp = pars["upp_bound"]
    x = np.stack((np.ones((num,)), np.linspace(low, upp, num))).T

    biao = sensitivity(x, best_pos, alpha, beta, link_type)

    plt.figure(figsize=(15, 12))
    plt.plot(np.linspace(low, upp, num), biao, c=pars["sensitivity_color"], linewidth=4)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("sensitivity", fontsize=20)
    plt.title(link_type + ": a=" + str(alpha) + " b=" + str(beta) +
              "\n" + "Design points: " + str(np.round(best_pos[:(d // 2)], 2)) +
              "\n" + "Probability: " + str(np.round(best_pos[(d // 2):], 2)), fontsize=24)
    plt.savefig(link_type + ", a=" + str(alpha) + ", b=" + str(beta) + ".png")

parser = argparse.ArgumentParser(description='2-point 2-parameter D-optimal design')

parser.add_argument('--file_path', type=str, default="pars.json", metavar='Path to arguments',
                    help="Path to arguments")

args = vars(parser.parse_args())

if __name__ == '__main__':

    optim_design(args["file_path"])