import networkx as nx
import pylab as pl

def scalar_validation_statistics(results, groups):
    """ plot absolute z transformation of p_theta values,
    grouped by dictionary groups

    Parameters
    ----------
    results : pandas.DataFrame with row called 'z'
    groups : list of lists of columns of results
    """
    pl.figure()

    width = max(pl.absolute(results.ix['z']))
    for row, (g_name, g) in enumerate(reversed(groups)):
        z = pl.absolute(results.ix['z', g].__array__())
        pl.plot([pl.mean(z)], [row], 'o', color='k', mec='k', mew=1)
        pl.plot(z, [row]*len(z), 'o', color='none', mec='k', mew=1)

        msg = 'p: %s' % ', '.join(['%.3f'%p for p in sorted(results.ix['p', g]*len(g))])
        #msg += 'MAE: %s' % str(
        pl.text(1.1*width, row, msg, va='center', fontsize='small')

    pl.yticks(range(len(groups)), ['%s %d' % (g_name, len(g)) for (g_name, g) in reversed(groups)], fontsize='large')
    pl.axis([-.05*width, width*1.05, -.5, len(groups)-.5])
    pl.xlabel(r'Absolute $z$-score of $p_\theta$ values', fontsize='large')

    pl.subplots_adjust(right=.5)
