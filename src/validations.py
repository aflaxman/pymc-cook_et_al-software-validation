""" Script to validate models through simulation"""
import pymc as mc

import data
import models

reload(data)
reload(models)

def validate_simple_model(n=[33, 21, 22, 22, 24, 11]):
    # generate data and model
    d = data.simple_hierarchical_data(n)
    m = models.simple_hierarchical_model(d['y'])

    # fit model with MCMC
    #mc.MCMC(m).sample(20000, 10000, 2)
    mc.MCMC(m).sample(2000, 1000, 2)

    # tally posterior quantiles
    results = {}
    
    for var in 'inv_sigma_sq mu inv_tau_sq mu_by_tau alpha_bar alpha_bar_by_sigma'.split():
        stats = m[var].stats()
        results[var] = (d[var] > m[var].trace()).sum() / float(stats['n'])

    stats = m['alpha'].stats()
    for j, alpha_j in enumerate(d['alpha']):
        results['alpha_%d'%j] = (alpha_j > m['alpha'].trace()[:,j]).sum() / float(stats['n'])

    stats = m['alpha_by_sigma'].stats()
    for j, alpha_j_by_sigma in enumerate(d['alpha_by_sigma']):
        results['alpha_%d_by_sigma'%j] = (alpha_j_by_sigma > m['alpha_by_sigma'].trace()[:,j]).sum() / float(stats['n'])

    return results
