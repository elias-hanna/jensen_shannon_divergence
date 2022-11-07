import numpy as np
from multiprocessing import cpu_count
from scipy.spatial import cKDTree as KDTree

## Code adapted from https://mail.python.org/pipermail/scipy-user/2011-May/029521.html
## Added some quick fixes to prevent division by zero or other problems
## Open to better fixes

## Remark: this code uses sampled data distributions densities to estimate the KL/JS divergence
## The more samples, the more accurate the KL/JS divergence will be

def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
  continuous distributions IEEE International Symposium on Information
  Theory, 2008.
  """
  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)

  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  nn_x = xtree.query(x, k=2, eps=.01, p=2, workers=cpu_count()-1)
  nn_y = ytree.query(x, k=1, eps=.01, p=2, workers=cpu_count()-1)
  r = nn_x[0][:,1]
  s = nn_y[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.

  #### QUICK FIX ZONE; OPEN TO IMPROVEMENTS ####
  
  ## quick fix to prevent infs put them at 0 so they are ignored...
  r[r==np.inf] = 0
  s[s==np.inf] = 0
  ## quick fix to prevent division by zero
  ## while keeping the samples distance identical between the two distributions
  r += 0.0000000001
  s += 0.0000000001
  
  #### END OF QUICK FIX ZONE ####

  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def JSdivergence(x, y, verbose=False):
    """Compute the Jensen-Shannon divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Jensen-Shannon divergence D(P||Q).

    Based on KL divergence estiamtion from
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    
    n,d = x.shape
    m,dy = y.shape
    
    assert(d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)
    if verbose:
        print("Constructed x and y KDTree")
    
    ### Do x || m
    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    nn_x = xtree.query(x, k=2, eps=.01, p=2, workers=cpu_count()-1)
    nn_y = ytree.query(x, k=1, eps=.01, p=2, workers=cpu_count()-1)

    r = nn_x[0][:,1]
    s = nn_y[0]
    
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.

    #### QUICK FIX ZONE; OPEN TO IMPROVEMENTS ####
  
    ## quick fix to prevent infs put them at 0 so they are ignored...
    r[r==np.inf] = 0
    s[s==np.inf] = 0
    ## quick fix to prevent division by zero
    ## while keeping the samples distance identical between the two distributions
    r += 0.0000000001
    s += 0.0000000001
  
    #### END OF QUICK FIX ZONE ####

    mu = 1/2*(r+s)

    kl_x_m = -np.log(r/mu).sum() * d / n + np.log(m / (n - 1.))
    if verbose:
        print("Computed KL(x || m) estimation")

    ### Do y || m
    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    nn_x = xtree.query(y, k=1, eps=.01, p=2, workers=cpu_count()-1)
    nn_y = ytree.query(y, k=2, eps=.01, p=2, workers=cpu_count()-1)
    r = nn_x[0]
    s = nn_y[0][:,1]
    
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.

    #### QUICK FIX ZONE; OPEN TO IMPROVEMENTS ####
  
    ## quick fix to prevent infs put them at 0 so they are ignored...
    r[r==np.inf] = 0
    s[s==np.inf] = 0
    ## quick fix to prevent division by zero
    ## while keeping the samples distance identical between the two distributions
    r += 0.0000000001
    s += 0.0000000001
  
    #### END OF QUICK FIX ZONE ####

    mu = 1/2*(r+s)

    kl_y_m = -np.log(s/mu).sum() * d / n + np.log(m / (n - 1.))
    if verbose:
        print("Computed KL(y || m) estimation")
    
    return 1/2*(kl_y_m + kl_x_m)
