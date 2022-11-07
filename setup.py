# Created by Elias Hanna 
# Date: 7/11/22

from setuptools import setup, find_packages

setup(name='jsd',
      install_requires=['numpy', 'scipy'],
      version='1.0.0',
      packages=find_packages(),
      author="Elias Hanna",
      author_email="h.elias@hotmail.fr",
      description="Give functions that compute the Jensen-Shannon Divergence between two data distributions with same dimensions (but potentially different number of samples).",
)
