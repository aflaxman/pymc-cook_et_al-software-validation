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

    for row, g in enumerate(sorted(groups)):
        z = pl.absolute(results.ix['z', groups[g]].__array__())
        pl.plot([pl.mean(z)], [row], 'o', color='k', mec='k', mew=1)
        pl.plot(z, [row]*len(z), 'o', color='none', mec='k', mew=1)

    pl.yticks(range(len(groups)), ['%s %d' % (g, len(groups[g])) for g in sorted(groups)])
    width = max(pl.absolute(results.ix['z']))
    pl.axis([-.05*width, width*1.05, -.5, len(groups)-.5])
    pl.xlabel(r'Absolute $z$ transform of $p_\theta$ values')
