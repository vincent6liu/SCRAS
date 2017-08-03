import os
import sys
import shutil
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError('SCRAS requires Python 3')


setup(name='scrasGUI',
      version='0.0',
      description='single cell RNA analysis suite',
      author='',
      author_email='',
      package_dir={'': 'src'},
      packages=['magic', 'phenograph', 'scras'],
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
      scripts=['src/gui/scras_gui.py'],
)


# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))
