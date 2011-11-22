""" Class for wrangling data
"""

import pylab as pl
import pymc as mc

def simple_hierarchical_data(n):
    """ Generate data based on the simple one-way hierarchical model
    given in section 3.1.1::

        y[i,j] | alpha[j], sigma^2 ~ N(alpha[j], sigma^2) i = 1, ..., n_j, j = 1, ..., J;
        alpha[j] | mu, tau^2 ~ N(mu, tau^2) j = 1, ..., J.

        sigma^2 ~ Inv-Chi^2(5, 20)
        mu ~ N(5, 5^2)
        tau^2 ~ Inv-Chi^2(2, 10)

    Parameters
    ----------
    n : list, len(n) = J, n[j] = num observations in group j
    """

    inv_sigma_sq = mc.rgamma(alpha=2.5, beta=50.)
    mu = mc.rnormal(mu=5., tau=5.**-2.)
    inv_tau_sq = mc.rgamma(alpha=1., beta=10.)

    J = len(n)
    alpha = mc.rnormal(mu=mu, tau=inv_tau_sq, size=J)
    y = [mc.rnormal(mu=alpha[j], tau=inv_sigma_sq, size=n[j]) for j in range(J)]

    mu_by_tau = mu * pl.sqrt(inv_tau_sq)
    alpha_by_sigma = alpha * pl.sqrt(inv_sigma_sq)
    alpha_bar = alpha.sum()
    alpha_bar_by_sigma = alpha_bar * pl.sqrt(inv_sigma_sq)

    return vars()


    
