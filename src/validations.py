""" Script to validate models through simulation"""
import pylab as pl
import pymc as mc
import pandas
import scipy.stats.stats

import data
import models
import graphics

reload(data)
reload(models)
reload(graphics)

def good_simple_sim(n=[33, 21, 22, 22, 24, 11]):
    # generate data and model
    d = data.simple_hierarchical_data(n)
    m = models.simple_hierarchical_model(d['y'])

    # fit model with MCMC
    mc.MCMC(m).sample(2000, 1000)

    return d, m


def bad_simple_sim_1(n=[33, 21, 22, 22, 24, 11]):
    # generate data and model, intentionally misspecifying prior on mu
    d = data.simple_hierarchical_data(n)
    m = models.simple_hierarchical_model(d['y'])
    m['mu'].parents['mu'] = -5.
    m['mu'].parents['tau'] = .01**-2

    # fit model with MCMC
    mc.MCMC(m).sample(2000, 1000)

    return d, m


def bad_simple_sim_2(n=[33, 21, 22, 22, 24, 11]):
    # generate data and model
    d = data.simple_hierarchical_data(n)
    m = models.simple_hierarchical_model(d['y'])

    # fit model with MCMC, but with badly initialized step method
    mcmc = mc.MCMC(m)
    mcmc.use_step_method(mc.Metropolis, m['alpha'], proposal_sd=.0001)
    mcmc.sample(2000, 1000)

    return d, m


def validate_simple_model(N_rep=20, simulation=good_simple_sim):
    q = pandas.DataFrame()
    for n in range(N_rep):
        # simulate data and fit model
        d, m = simulation()

        # tally posterior quantiles
        results = {}

        for var in 'inv_sigma_sq mu inv_tau_sq mu_by_tau alpha_bar alpha_bar_by_sigma'.split():
            stats = m[var].stats()
            results[var] = [(d[var] > m[var].trace()).sum() / float(stats['n'])]

        stats = m['alpha'].stats()
        for j, alpha_j in enumerate(d['alpha']):
            results['alpha_%d'%j] = [(alpha_j > m['alpha'].trace()[:,j]).sum() / float(stats['n'])]

        stats = m['alpha_by_sigma'].stats()
        for j, alpha_j_by_sigma in enumerate(d['alpha_by_sigma']):
            results['alpha_%d_by_sigma'%j] = [(alpha_j_by_sigma > m['alpha_by_sigma'].trace()[:,j]).sum() / float(stats['n'])]

        q = q.append(pandas.DataFrame(results, index=['q_rep_%d'%n]))

    z = {}
    for var in q.columns:
        X_var = pl.sum(mc.utils.invcdf(q[var])**2)
        p_var = scipy.stats.stats.chisqprob(X_var, N_rep)
        z[var] = [mc.utils.invcdf(p_var)]
    
    results = pandas.DataFrame(z, index=['z']).append(q)

    graphics.scalar_validation_statistics(
        results, 
        {r'$\mu/\tau$': ['mu_by_tau'],
         r'$\sigma^{-2}$': ['inv_sigma_sq'],
         r'$\tau^{-2}$': ['inv_tau_sq'],
         r'$\mu$': ['mu'],
         r'$\alpha/\sigma$': results.filter(regex='alpha_\d_by_sigma').columns,
         r'$\alpha$': results.filter(regex='alpha_\d$').columns})

    return results
