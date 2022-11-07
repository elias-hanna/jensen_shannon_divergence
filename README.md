# Jensen Shannon Divergence for continous data distributions
## General remarks
This repo contains code (in jsd.py) that will compute the Kullback-Leibler divergence or the Jensen-Shannon divergence between two continuous data distributions.\
Code has been adapted from the KL implementation from https://mail.python.org/pipermail/scipy-user/2011-May/029521.html and extended to provide JS divergence as well.\ 
KL estimation method is taken from Pérez-Cruz, F. Kullback-Leibler divergence estimation of continuous distributions IEEE International Symposium on Information Theory, 2008.\
If someone has time and some ideas on how to do so properly, it would be great to compare results with scipy.spatial.distance.jensenshannon (which takes as inputs probability vectors) with the results obtained here (with data distributions directly).\

## How to use
`git clone https://github.com/elias-hanna/jensen_shannon_divergence`\
`cd jensen_shannon_divergence && pip install -e .`\
Now, from any file in your python environment where you installed the jsd module, you can import the KL/JS divergence like this: `from jsd import KLdivergence` or `from jsd import JSdivergence`.
These functions takes as args both data distributions you want to compute the KL/JS divergence for. It's important to note that the order of arguments has an impact on the divergence computed for KL.
