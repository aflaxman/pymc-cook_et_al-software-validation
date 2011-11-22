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


def complex_hierarchical_data(n):
    """ Generate data based on the much more complicated model
    given in section 3.2.1::

        y_ij ~ N(mu_j - exp(beta_j)t_ij - exp(gamma_j)t_ij^2, sigma_j^2)
        gamma_j | sigma^2, xi, X_j ~ N(eta_0 + eta_1 X_j + eta_2 X_j^2, omega^2)
        beta_j | gamma_j, sigma^2, xi, X_j ~ N(delta_beta_0 + delta_beta_1 X_j + delta_beta_2 X_j^2 + delta_beta_3 gamma_j, omega_beta^2)
        mu_j | gamma_j, beta_j, sigma^2, xi, X_j ~ N(delta_mu_0 + delta_mu_1 X_j + delta_mu_2 X_j^2 + delta_mu_3 gamma_j + delta_mu_4 beta_j, omega_mu^2)

        eta = (eta_0, eta_1, eta_2, log(omega))'
        delta_beta = (delta_beta_0, delta_beta_1, delta_beta_2, delta_beta_3, log(omega_beta))'
        delta_mu = (delta_mu_0, delta_mu_1, delta_mu_2, delta_mu_3, log(omega_mu))'
        xi = (eta, delta_beta, delta_mu)
        eta ~ MVNormal(M, C)
        delta_beta, delta_mu ~ Normal(m, s)

    Parameters
    ----------
    n : list, len(n) = J, n[j] = num observations in group j
    """

    J = len(n)
    
    # covariate data, not entirely specified in paper
    X = mc.rnormal(0, 1, size=J)
    t = [pl.arange(n[j]) for j in range(J)]

    # hyper-priors, not specified in detail in paper
    m = 0.
    s = 1.
    M = pl.zeros(4)
    r = [[  1, .57, .18, .56],
         [.57,   1, .72, .16],
         [.18, .72,   1, .14],
         [.56, .16, .14,   1]]

    eta = mc.rmv_normal_cov(M, r)
    omega = pl.exp(eta[-1])

    delta_beta = mc.rnormal(m, s**-2, size=5)
    omega_beta = pl.exp(delta_beta[-1])

    delta_mu = mc.rnormal(m, s**-2, size=5)
    omega_mu = pl.exp(delta_mu[-1])

    gamma = mc.rnormal(eta[0] + eta[1]*X + eta[2]*X**2, omega**-2.)
    beta = mc.rnormal(delta_beta[0] + delta_beta[1]*X + delta_beta[2]*X**2 + delta_beta[3]*gamma, omega_beta**-2)
    mu = mc.rnormal(delta_mu[0] + delta_mu[1]*X + delta_mu[2]*X**2 + delta_mu[3]*gamma + delta_mu[4]*beta, omega_mu**-2)

    # stochastic error, not specified in paper
    sigma = pl.ones(J)
    y = [mc.rnormal(mu[j] - pl.exp(beta[j])*t[j] - pl.exp(gamma[j])*t[j]**2, sigma[j]**-2) for j in range(J)]

    eta_cross_eta = [eta[0]*eta[1], eta[0]*eta[2], eta[0]*eta[3], eta[1]*eta[2], eta[1]*eta[2], eta[2]*eta[3]]

    return vars()
