from shiny import App, reactive, render, ui, req
import shinyswatch
import shiny.experimental as x

from htmltools import css
import pandas as pd
import numpy as np
import numpy.matlib as m
import polars as pl

from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

from scipy.stats import norm

normal = norm()

p1 = norm.rvs(10, 1, 1000)
p2 = norm.rvs(10.5, 1, 1000)

hist1, bins1 = np.histogram(p1, bins=30, density=True)


plt.hist(p1, bins=30, density=True, alpha=0.5, color="b")
plt.show()

combined = np.concatenate([p1, p2])
permutation_results = []

for _ in range(10_000):
    combined = np.random.permutation(combined)
    perm_p1 = combined[: len(p1)]
    perm_p2 = combined[-len(p1) :]
    permutation_results.append(np.mean(perm_p1) - np.mean(perm_p2))

plt.hist(permutation_results, bins=30, density=True, alpha=0.5, color="b")
plt.show()

np.mean(np.array(permutation_results) >= 0.1)

