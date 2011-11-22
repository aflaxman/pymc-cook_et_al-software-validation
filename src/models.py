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


def complex_hierarchical_model(y, X, t):
    """ PyMC model for the complicated example given in section
    3.2.1::

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
    y : list, len(y) = J, y[j][i] = measurement i on patient j
    X : list, len(X) = J, X[j] = baseline measurement for patient j
    t : list, len(t) = J, t[j][i] = time of measurement i on patient j
    """

    J = len(y)
    
    # hyper-priors, not specified in detail in paper
    m = 0.
    s = 1.
    M = pl.zeros(4)
    C = pl.ones((4,4)) + pl.eye(4)

    eta = mc.MvNormalCov('eta', M, C)
    omega = mc.Lambda('omega', lambda eta=eta: pl.exp(eta[-1]))
    
    delta_beta = mc.Normal('delta_beta', m, s**-2, size=5)
    omega_beta = mc.Lambda('omega_beta', lambda delta_beta=delta_beta: pl.exp(delta_beta[-1]))

    delta_mu = mc.Normal('delta_mu', m, s**-2, size=5)
    omega_mu = mc.Lambda('omega_mu', lambda delta_mu=delta_mu: pl.exp(delta_mu[-1]))

    gamma = mc.Normal('gamma', eta[0] + eta[1]*X + eta[2]*X**2, omega**-2.)
    beta = mc.Normal('beta', delta_beta[0] + delta_beta[1]*X + delta_beta[2]*X**2 + delta_beta[3]*gamma, omega_beta**-2)
    mu = mc.Normal('mu', delta_mu[0] + delta_mu[1]*X + delta_mu[2]*X**2 + delta_mu[3]*gamma + delta_mu[4]*beta, omega_mu**-2)

    sigma = mc.Uniform('sigma', 0., 10., value=pl.ones(J))
    y_exp = [mc.Lambda('y_exp_%d'%j, lambda mu=mu, beta=beta, gamma=gamma, j=j: mu[j] - pl.exp(beta[j])*t[j] - pl.exp(gamma[j])*t[j]**2) for j in range(J)]
    @mc.potential
    def y_obs(y_exp=y_exp, sigma=sigma, y=y):
        logp = 0.
        for j in range(J):
            missing = pl.isnan(y[j])
            logp += mc.normal_like(y[j][~missing], y_exp[j][~missing], sigma[j]**-2)
        return logp

    y_pred = [mc.Normal('y_pred_%d'%j, y_exp[j], sigma[j]**-2) for j in range(J)]

    eta_cross_eta = mc.Lambda('eta_cross_eta', lambda eta=eta: [eta[0]*eta[1], eta[0]*eta[2], eta[0]*eta[3], eta[1]*eta[2], eta[1]*eta[2], eta[2]*eta[3]])

    return vars()
