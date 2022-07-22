import datetime as dt
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fw_blasso import VanillaFWforBLasso
from simtools import NUFSamplingCstrctr
from utils import DiracStream

import pycsou.abc.solver as pycs
import pycsou.opt.stop as pycos

matplotlib.use("qt5agg")

# By convention, in Pycsou the stacking of data is done as/according to (..., input_dim)
# data[0, :] is the first element

##############################################

lambda_factor = 0.1

n_sources = 10
L = 100
psnr = 100
T = 2  # Side of the square area
r = 0.8  # sampling area (rate of the side length)
plot_grid_size = 256
seed = None
plot_examples = False

##############################################

if seed is None:
    seed = np.random.randint(0, 10000)
print("Seed: ", seed)

rng = np.random.default_rng(seed)

xmin = -T / 2
xmax = T / 2
ymin = -T / 2
ymax = T / 2
support = np.array([[xmin, xmax], [ymin, ymax]])
T_s = r * T
xmin_s = r * xmin
ymin_s = r * ymin

## Sources
sources_pos = np.stack(
    [rng.random(size=n_sources) * T_s + xmin_s, rng.random(size=n_sources) * T_s + ymin_s]
).transpose()  # shape (n_sources, 2)
sources_weights = rng.uniform(2.0, 6.0, size=n_sources)  # rng.chisquare(1, n_sources)
# 2D sources
sources = DiracStream(positions=sources_pos, weights=sources_weights, support_width=np.array([T, T]))

if plot_examples:
    plt.figure(figsize=(7, 6))
    plt.scatter(
        sources.positions[:, 0],
        sources.positions[:, 1],
        s=100 * sources.weights / sources.weights.max(),
        label="sources",
        marker="o",
        c="r",
        alpha=0.8,
    )
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    # plt.plot([xmin_s, xmin_s + T_s, xmin_s + T_s, xmin_s, xmin_s], [ymin_s, ymin_s, ymin_s + T_s, ymin_s + T_s, ymin_s],
    #         alpha=.5, label='border')
    plt.legend(loc=2)
    plt.title("2D Stream of Dirac Impulses")
    plt.show()

## Baselines
# sigma_psf = T/6
sigma_psf = T / 100  # max(d_min, T/50)
sampling_cov_mat = 1 / (2 * np.pi * sigma_psf) * np.eye(2)
baselines = rng.multivariate_normal(np.zeros(2), sampling_cov_mat, size=L)  # shape (L, 2)

if plot_examples:
    plt.figure(figsize=(7, 6))
    plt.scatter(baselines[:, 0], baselines[:, 1], marker="+")
    plt.title("Distribution of the sampled frequencies")
    plt.show()

## Measurements
phi_cstrctr = NUFSamplingCstrctr(frequencies=baselines, support_width=np.array([T, T]))
gt_forward = phi_cstrctr.fixedKnotsForwardOp(sources_pos)
gt_visibilities = gt_forward(sources_weights)
variance = np.max(gt_visibilities[:L] ** 2 + gt_visibilities[L:] ** 2) * 10 ** (-psnr / 10)
std = np.sqrt(variance / 2)
noise = rng.normal(0, std, size=2 * L)
measurements = gt_visibilities + noise

if plot_examples:
    x = np.linspace(xmin, xmax, plot_grid_size)
    y = np.linspace(ymin, ymax, plot_grid_size)
    flatten_grid = np.stack(np.meshgrid(x, y)).reshape((2, -1)).transpose()
    dirty_image = phi_cstrctr.fixedEvaluationPointsAdjointOp(flatten_grid)(measurements)
    plt.figure(figsize=(7, 6))
    # plt.imshow(np.flip(dirty_image.reshape((plot_grid_size, plot_grid_size)), axis=0))
    plt.imshow(dirty_image.reshape((plot_grid_size, plot_grid_size)), origin="lower")
    plt.scatter(
        ((sources_pos - xmin) * plot_grid_size / T)[:, 0],
        (((sources_pos - ymin) / T) * plot_grid_size)[:, 1],
        marker="+",
        color="r",
    )
    plt.colorbar()
    plt.title("Dirty Image and Point Sources")
    plt.show()

## Solving

# random sampling based estimation of lambda
random_samples = np.random.uniform(-T / 2, T / 2, size=(3000, 2))
adj_lambda = phi_cstrctr.fixedEvaluationPointsAdjointOp(random_samples)
evaluations = adj_lambda(measurements)
lambda_ = lambda_factor * np.max(np.abs(evaluations))

"""
grid_vfw = VanillaFWforBLasso(data=measurements,
                              forwardOpCstrctr=phi_cstrctr,
                              lambda_=lambda_,
                              verbosity=20,
                              optim_strategy="stochastic",
                              step_size="line_search")

print("\nVanilla FW: Solving ...")
start = time.time()
grid_vfw.fit()
t_v = time.time() - start
data_v, hist_v = grid_vfw.stats()
print("\tSolved in {:.3f} seconds".format(t_v))

plt.figure(figsize=(7, 6))
plt.scatter(
    sources.positions[:, 0],
    sources.positions[:, 1],
    s=100 * sources.weights / sources.weights.max(),
    label="sources",
    marker="o",
    c="r",
    alpha=0.8,
)
plt.scatter(
    data_v["positions"][:, 0],
    data_v["positions"][:, 1],
    s=100 * data_v["weights"] / sources.weights.max(),
    label="Reconstruction",
    marker="o",
    c="b",
    alpha=0.8,
)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
# plt.plot([xmin_s, xmin_s + T_s, xmin_s + T_s, xmin_s, xmin_s], [ymin_s, ymin_s, ymin_s + T_s, ymin_s + T_s, ymin_s],
#         alpha=.5, label='border')
plt.legend(loc=2)
plt.title("2D Stream of Dirac Impulses")
plt.show()
"""

#################################

optims = ["grid", "stochastic"]
steps = ["regular", "line_search"]
solvers = []
times = []
solutions = []
for o in optims:
    for s in steps:
        solvers.append(
            VanillaFWforBLasso(
                data=measurements,
                forwardOpCstrctr=phi_cstrctr,
                lambda_=lambda_,
                verbosity=20,
                optim_strategy=o,
                step_size=s,
            )
        )

max_duration = pycos.MaxDuration(dt.timedelta(seconds=20))
stop_crit = pycos.RelError(
    eps=1e-32,
    var="ofv",
    f=None,
    norm=2,
    satisfy_all=True,
)

for i, s in enumerate(solvers):
    print(f"Vanilla FW: Solver {i}")
    start = time.time()
    s.fit(stop_crit=max_duration | stop_crit)
    times.append(time.time() - start)
    data_sol, _ = s.stats()
    solutions.append(data_sol)

plt.figure(figsize=(13, 13))
for i, data in enumerate(solutions):
    plt.subplot(2, 2, i + 1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.scatter(
        sources.positions[:, 0],
        sources.positions[:, 1],
        s=100 * sources.weights / sources.weights.max(),
        label="sources",
        marker="o",
        c="r",
        alpha=0.8,
    )
    plt.scatter(
        data["positions"][:, 0][data["weights"] > 0],
        data["positions"][:, 1][data["weights"] > 0],
        s=100 * data["weights"][data["weights"] > 0] / sources.weights.max(),
        label="Reconstruction",
        marker="o",
        c="b",
        alpha=0.6,
    )
    plt.scatter(
        data["positions"][:, 0][data["weights"] < 0],
        data["positions"][:, 1][data["weights"] < 0],
        s=-100 * data["weights"][data["weights"] < 0] / sources.weights.max(),
        label="Reconstruction",
        marker="o",
        c="green",
        alpha=0.6,
    )
    plt.title(optims[i // 2] + " " + steps[i % 2] + ": {:.3f} - OFV: {:.1f}".format(times[i], data["ofv"][0]))
plt.show()
