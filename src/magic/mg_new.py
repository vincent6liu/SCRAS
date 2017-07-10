import re
import os
import random
import pickle
import warnings
import shlex
import shutil
from copy import deepcopy
from collections import defaultdict, Counter
from subprocess import call, Popen, PIPE
import glob

import numpy as np
import pandas as pd

import matplotlib
# try:
#     os.environ['DISPLAY']
# except KeyError:
#     matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # catch experimental ipython widget warning
    import seaborn as sns
from matplotlib.font_manager import FontProperties

# from tsne import bh_sne
from sklearn.manifold import TSNE
from sklearn.manifold.t_sne import _joint_probabilities, _joint_probabilities_nn
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, find, vstack, hstack, issparse, coo
from scipy.sparse.linalg import eigs
from numpy.linalg import norm
from scipy.stats import gaussian_kde
from scipy.io import mmread
from numpy.core.umath_tests import inner1d
import fcsparser
import magic
import phenograph

# set plotting defaults
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # catch experimental ipython widget warning
    sns.set(context="paper", style='ticks', font_scale=1.5, font='Bitstream Vera Sans')

matplotlib.rcParams['image.cmap'] = 'viridis'
size = 12


def qualitative_colors(n):
    """ Generalte list of colors
    :param n: Number of colors
    """
    return sns.color_palette('Set1', n)


def get_fig(fig=None, ax=None, figsize=[6.5, 6.5]):
    """fills in any missing axis or figure with the currently active one
    :param ax: matplotlib Axis object
    :param fig: matplotlib Figure object
    """
    if not fig:
        fig = plt.figure(figsize=figsize)
    if not ax:
        ax = plt.gca()
    return fig, ax


def density_2d(x, y):
    """return x and y and their density z, sorted by their density (smallest to largest)

    :param x:
    :param y:
    :return:
    """
    xy = np.vstack([np.ravel(x), np.ravel(y)])
    z = gaussian_kde(xy)(xy)
    i = np.argsort(z)
    return np.ravel(x)[i], np.ravel(y)[i], np.arcsinh(z[i])


class SCData:
    def __init__(self, name: str, data, data_type='sc-seq', metdata=None):
        if not (isinstance(data, pd.DataFrame)):
            raise TypeError('data must be of type DataFrame')
        if data_type not in ['sc-seq', 'masscyt']:
            raise RuntimeError('data type must be either sc-seq or masscyt')
        if metdata is None:
            metadata = pd.DataFrame(index=data.index, dtype='0')

        # initiate the data dictionary with the given data
        cols = [np.array(['data'] * data.shape[1]), np.array(data.columns.values)]
        self._data = {name: pd.DataFrame(data.values, index=data.index, columns=cols)}
        self._main = self._data[name]
        self._name = name
        self._metadata = metadata
        self._data_type = data_type
        self._normalized = False
        self._logtrans = False
        self._magic = False

        # Library size (whats this??)
        self._library_sizes = None

    # may need to be rewritten
    def reset(self):
        self._normalized = False
        self._logtrans = False
        self._magic = False
        self._library_sizes = None
        self._backup = None

    def save(self, out_file: str):
        with open(out_file, 'wb') as f:
            pickle.dump(vars(self), f)

    @classmethod
    def load(cls, in_file: str):
        with open(in_file, 'rb') as f:
            data = pickle.load(f)
        scdata = cls(data['_name'], data['_data'], data['_metadata'])
        del data['_name']
        del data['_data']
        del data['_metadata']
        for k, v in data.items():
            setattr(scdata, k[1:], v)
        return scdata

    def __repr__(self):
        c, g = self.data[self.name].shape
        _repr = ('SCData: {c} cells x {g} genes\n'.format(g=g, c=c))
        for k, v in sorted(vars(self).items()):
            if not (k == '_data'):
                _repr += '\n{}={]'.format(k[1:], 'None' if v is None else 'True')
        return _repr

    @property
    def name(self):
        return self._name

    @property
    def data_type(self):
        return self._data_type

    @property
    def logtrans(self):
        return self._logtrans

    @property
    def magic(self):
        return self._magic

    @property
    def normalized(self):
        return self._normalized

    @property
    def datadict(self):
        return self._data

    @property
    def data(self):
        return self._main

    @data.setter
    def data(self, item):
        if not (isinstance(item, pd.DataFrame)):
            raise TypeError('SCData.data must be of type DataFrame')
        cols = [np.array(['data'] * item.shape[1]), np.array(item.columns.values)]
        self._data = {self.name: pd.DataFrame(item.values, index=item.index, columns=cols)}
        self._main = pd.DataFrame(item.values, index=item.index, columns=cols)
        self.reset()

    @property
    def extended_data(self):
        return self._data

    @extended_data.setter
    def extended_data(self, item):
        if not (isinstance(item, pd.DataFrame)):
            raise TypeError('SCData.extended_data must be of type DataFrame')
        self._data = item

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, item):
        if not isinstance(item, pd.DataFrame):
            raise TypeError('SCData.metadata must be of type DataFrame')
        self._metadata = item

    @property
    def library_sizes(self):
        return self._library_sizes

    @library_sizes.setter
    def library_sizes(self, item):
        if not (isinstance(item, pd.Series) or item is None):
            raise TypeError('self.library_sizes must be a pd.Series object')
        self._library_sizes = item

    @classmethod
    def from_csv(cls, counts_csv_file, data_name: str, data_type='sc-seq', cell_axis=0, delimiter=',',
                 rows_after_header_to_skip=0, cols_after_header_to_skip=0, normalize=True):

        if not data_type in ['sc-seq', 'masscyt']:
            raise RuntimeError('data_type must be either sc-seq or masscyt')

        # Read in csv file
        df = pd.DataFrame.from_csv(counts_csv_file, sep=delimiter)

        df.drop(df.index[1:rows_after_header_to_skip + 1], axis=0, inplace=True)
        df.drop(df.columns[1:cols_after_header_to_skip + 1], axis=1, inplace=True)

        if cell_axis != 0:
            df = df.transpose()

        # Construct class object
        scdata = cls(data_name, df, data_type=data_type)

        # Normalize if specified
        if normalize:
            scdata = scdata.normalize_scseq_data()

        return scdata

    @classmethod
    def from_fcs(cls, fcs_file, data_name: str, cofactor=5,
                 metadata_channels=['Time', 'Event_length', 'DNA1', 'DNA2', 'Cisplatin', 'beadDist', 'bead1']):

        # Parse the fcs file
        text, data = fcsparser.parse(fcs_file)
        data = data.astype(np.float64)

        # Extract the S and N features (Indexing assumed to start from 1)
        # Assumes channel names are in S
        no_channels = text['$PAR']
        channel_names = [''] * no_channels
        for i in range(1, no_channels + 1):
            # S name
            try:
                channel_names[i - 1] = text['$P%dS' % i]
            except KeyError:
                channel_names[i - 1] = text['$P%dN' % i]
        data.columns = channel_names

        # Metadata and data
        metadata_channels = data.columns.intersection(metadata_channels)
        data_channels = data.columns.difference(metadata_channels)
        metadata = data[metadata_channels]
        data = data[data_channels]

        # Transform if necessary
        if cofactor is not None or cofactor > 0:
            data = np.arcsinh(np.divide(data, cofactor))

        # Create and return scdata object
        scdata = cls(data_name, data, 'masscyt', metadata)
        return scdata

    @classmethod
    def from_mtx(cls, mtx_file, gene_name_file, data_name: str, normalize=True):

        # Read in mtx file
        count_matrix = mmread(mtx_file)

        gene_names = np.loadtxt(gene_name_file, dtype=np.dtype('S'))
        gene_names = np.array([gene.decode('utf-8') for gene in gene_names])

        ### remove todense
        df = pd.DataFrame(count_matrix.todense(), columns=gene_names)

        # Construct class object
        scdata = cls(data_name, df, data_type='sc-seq')

        # Normalize if specified
        if normalize:
            scdata = scdata.normalize_scseq_data()

        return scdata

    @classmethod
    def from_10x(cls, data_dir, data_name: str, use_ensemble_id=True, normalize=True):
        # loads 10x sparse format data
        # data_dir is dir that contains matrix.mtx, genes.tsv and barcodes.tsv
        # return_sparse=True -- returns data matrix in sparse format (default = False)

        if not data_dir or not len(data_dir):
            data_dir = './'
        elif data_dir[len(data_dir) - 1] != '/':
            data_dir = data_dir + '/'

        filename_dataMatrix = os.path.expanduser(data_dir + 'matrix.mtx')
        filename_genes = os.path.expanduser(data_dir + 'genes.tsv')
        filename_cells = os.path.expanduser(data_dir + 'barcodes.tsv')

        # Read in gene expression matrix (sparse matrix)
        # Rows = genes, columns = cells
        print('LOADING')
        dataMatrix = mmread(filename_dataMatrix);

        # Read in row names (gene names / IDs)
        gene_names = np.loadtxt(filename_genes, delimiter='\t', dtype=bytes).astype(str)
        if use_ensemble_id:
            gene_names = [gene[0] for gene in gene_names]
        else:
            gene_names = [gene[1] for gene in gene_names]
        cell_names = np.loadtxt(filename_cells, delimiter='\t', dtype=bytes).astype(str)

        dataMatrix = pd.DataFrame(dataMatrix.todense(), columns=cell_names, index=gene_names)

        # combine duplicate genes
        if not use_ensemble_id:
            dataMatrix = dataMatrix.groupby(dataMatrix.index).sum()

        dataMatrix = dataMatrix.transpose()

        # Remove empty cells
        print('Removing empty cells')
        cell_sums = dataMatrix.sum(axis=1)
        to_keep = np.where(cell_sums > 0)[0]
        dataMatrix = dataMatrix.ix[dataMatrix.index[to_keep], :].astype(np.float32)

        # Remove empty genes
        print('Removing empty genes')
        gene_sums = dataMatrix.sum(axis=0)
        to_keep = np.where(gene_sums > 0)[0]
        dataMatrix = dataMatrix.ix[:, to_keep].astype(np.float32)

        # Construct class object
        scdata = cls(data_name, dataMatrix, data_type='sc-seq')

        # Normalize if specified
        if normalize:
            scdata = scdata.normalize_scseq_data()

        return scdata

    def filter_scseq_data(self, filter_cell_min=0, filter_cell_max=0, filter_gene_nonzero=None, filter_gene_mols=None):

        if filter_cell_min != filter_cell_max:
            sums = self.data.sum(axis=1)
            to_keep = np.intersect1d(np.where(sums >= filter_cell_min)[0],
                                     np.where(sums <= filter_cell_max)[0])
            self.data = self.data.ix[self.data.index[to_keep], :].astype(np.float32)

        if filter_gene_nonzero:
            nonzero = self.data.astype(bool).sum(axis=0)
            to_keep = np.where(nonzero >= filter_gene_nonzero)[0]
            self.data = self.data.ix[:, to_keep].astype(np.float32)

        if filter_gene_mols:
            sums = self.data.sum(axis=0)
            to_keep = np.where(sums >= filter_gene_mols)[0]
            self.data = self.data.ix[:, to_keep].astype(np.float32)

    def normalize_scseq_data(self):
        """
        Normalize single cell RNA-seq data: Divide each cell by its molecule count
        and multiply counts of cells by the median of the molecule counts
        :return: SCData
        """

        molecule_counts = self.data.sum(axis=1)
        data = self.data.div(molecule_counts, axis=0) \
            .mul(np.median(molecule_counts), axis=0)
        scdata = SCData(name=self.name, data=data, metadata=self.metadata)
        scdata._normalized = True

        # check that none of the genes are empty; if so remove them
        nonzero_genes = scdata.data.sum(axis=0) != 0
        scdata.data = scdata.data.ix[:, nonzero_genes].astype(np.float32)

        # set unnormalized_cell_sums
        self.library_sizes = molecule_counts
        scdata._library_sizes = molecule_counts

        return scdata

    def log_transform_scseq_data(self, pseudocount=0.1):
        key = self._name + ":logtrans"
        self._data[key] = np.log(np.add(self.data, pseudocount))
        self._backup = self._main
        self._main = self._data[key]
        self._logtrans = True

    def plot_molecules_per_cell_and_gene(self, fig=None, ax=None):
        height = 4
        width = 12
        if not fig:
            fig = plt.figure(figsize=[width, height])
        gs = plt.GridSpec(1, 3)
        colsum = np.log10(self.data.sum(axis=0))
        rowsum = np.log10(self.data.sum(axis=1))
        for i in range(3):
            ax = plt.subplot(gs[0, i])
            if not i:
                print(np.min(rowsum))
                print(np.max(rowsum))
                ax.hist(rowsum, bins='auto')
                plt.xlabel('Molecules per cell (log10 scale)')
            elif i == 1:
                temp = np.log10(self.data.astype(bool).sum(axis=0))
                ax.hist(temp, bins='auto')
                plt.xlabel('Nonzero cells per gene (log10 scale)')
            else:
                ax.hist(colsum, bins='auto')
                plt.xlabel('Molecules per gene (log10 scale)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            ax.tick_params(axis='x', labelsize=8)

        return fig, ax

    def run_pca(self, n_components=100, random=True):
        """
        Principal component analysis of the data.
        :param n_components: Number of components to project the data
        """
        key = self.name + ":PCA:" + str(n_components)
        solver = 'randomized'
        if random != True:
            solver = 'full'

        pca = PCA(n_components=n_components, svd_solver=solver)
        self.datadict[key] = pd.DataFrame(data=pca.fit_transform(self.data.values), index=self.data.index,
                                          columns=['PC' + str(i) for i in range(1, n_components + 1)])

    def plot_pca_variance_explained(self, n_components=30,
                                    fig=None, ax=None, ylim=(0, 100), random=True):
        """ Plot the variance explained by different principal components
        :param n_components: Number of components to show the variance
        :param ylim: y-axis limits
        :param fig: matplotlib Figure object
        :param ax: matplotlib Axis object
        :return: fig, ax
        """

        solver = 'randomized'
        if random != True:
            solver = 'full'
        pca = PCA(n_components=n_components, svd_solver=solver)
        pca.fit(self.data.values)

        fig, ax = get_fig(fig=fig, ax=ax)
        plt.plot(np.multiply(np.cumsum(pca.explained_variance_ratio_), 100))
        plt.ylim(ylim)
        plt.xlim((0, n_components))
        plt.xlabel('Components')
        plt.ylabel('Percent Variance explained')
        plt.title('Principal components')
        return fig, ax
