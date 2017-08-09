import os
import sys
from setuptools import setup

if sys.version_info.major != 3:
    raise RuntimeError('SCRAS requires Python 3')


setup(name='scras',
      version='0.0',
      description='Single cell RNA analysis suite',
      author='Vincent Liu',
      author_email='vincent.liu@columbia.edu',
      package_dir={'': 'src'},
      packages=['magic', 'phenograph'],
      package_data={
          'phenograph': ['louvain/*convert*', 'louvain/*community*', 'louvain/*hierarchy*']},
      install_requires=[
          'numpy>=1.10.0',
          'pandas>=0.18.0',
          'scipy>=0.14.0',
          'matplotlib',
          'seaborn',
          'sklearn',
          'networkx',
          'fcsparser',
          'statsmodels',
      ],
      scripts=['src/scras/scras_gui.py'],
    )


# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))
