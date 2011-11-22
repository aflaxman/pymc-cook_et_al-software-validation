""" Module for setting up statistical models
"""

import pylab as pl
import pymc as mc

def simple_hierarchical_model(y):
    """ PyMC implementation of the simple hierarchical model from
    section 3.1.1::

        y[i,j] | alpha[j], sigma^2 ~ N(alpha[j], sigma^2) i = 1, ..., n_j, j = 1, ..., J;
        alpha[j] | mu, tau^2 ~ N(mu, tau^2) j = 1, ..., J.

        sigma^2 ~ Inv-Chi^2(5, 20)
        mu ~ N(5, 5^2)
        tau^2 ~ Inv-Chi^2(2, 10)

    Parameters
    ----------
    y : a list of lists of observed data, len(y) = J, len(y[j]) = n_j
    """

    inv_sigma_sq = mc.Gamma('inv_sigma_sq', alpha=2.5, beta=50.)
    mu = mc.Normal('mu', mu=5., tau=5.**-2.)
    inv_tau_sq = mc.Gamma('inv_tau_sq', alpha=1., beta=10.)

    J = len(y)
    alpha = mc.Normal('alpha', mu=mu, tau=inv_tau_sq, size=J)
    y = [mc.Normal('y_%d'%j, mu=alpha[j], tau=inv_sigma_sq, value=y[j], observed=True) for j in range(J)]

    @mc.deterministic
    def mu_by_tau(mu=mu, tau=inv_tau_sq**-.5):
        return mu/tau

    @mc.deterministic
    def alpha_by_sigma(alpha=alpha, sigma=inv_sigma_sq**-.5):
        return alpha/sigma

    alpha_bar = mc.Lambda('alpha_bar', lambda alpha=alpha: pl.sum(alpha))

    @mc.deterministic
    def alpha_bar_by_sigma(alpha_bar=alpha_bar, sigma=inv_sigma_sq**-.5):
        return alpha_bar/sigma

    return vars()
