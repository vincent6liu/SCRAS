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
        self._datadict = {name: pd.DataFrame(data.values, index=data.index, columns=cols)}
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

    # returns the data dictionary
    @property
    def datadict(self):
        return self._datadict

    # returns the raw or normalized data
    @property
    def data(self):
        return self._datadict[self.name]

    @data.setter
    def data(self, item):
        if not (isinstance(item, pd.DataFrame)):
            raise TypeError('SCData.data must be of type DataFrame')
        cols = [np.array(['data'] * item.shape[1]), np.array(item.columns.values)]
        self._datadict = {self.name: pd.DataFrame(item.values, index=item.index, columns=cols)}
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
        self.datadict[key] = np.log(np.add(self.data, pseudocount))
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

    def run_pca(self, n_components=100, rand=True):
        """
        Principal component analysis of the data.
        :param n_components: Number of components to project the data
        :param rand: Whether randomized
        """
        key = self.name + ":PCA:" + str(n_components)
        solver = 'randomized' if rand else 'full'

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

    def run_tsne(self, n_components=50, perplexity=30, n_iter=1000, theta=0.5):
        """ Run tSNE on the data. tSNE is run on the principal component projections
        for single cell RNA-seq data and on the expression matrix for mass cytometry data
        stored in the data dictionary as "name:TSNE:comp-perp-iter-theta"
        :param n_components: Number of components to use for running tSNE for single cell
        RNA-seq data. Ignored for mass cytometry
        :return: None
        """
        pca_keys = [pca_key for pca_key in self.datadict.keys() if 'PCA' in pca_key.upper()]
        if bool(pca_keys):
            comps = sorted([int(key[key.rfind(':')+1:]) for key in pca_keys])
            low_comp = min(comps)

        # Work on PCA projections if data is single cell RNA-seq
        if self.data_type == 'sc-seq':
            if n_components in comps:
                data = self.datadict[(self.name+":PCA:"+str(n_components))]
            elif (not bool(pca_keys)) or n_components > low_comp:
                self.run_pca(n_components=n_components)
                data = self.datadict[(self.name+":PCA:"+str(n_components))]
            else:  # n_components <= low_comp
                data = self.datadict[(self.name+":PCA:"+str(low_comp))].iloc[:, :n_components]
        else:
            data = self.data

        # Reduce perplexity if necessary
        perplexity_limit = 15
        if data.shape[0] < 100 and perplexity > perplexity_limit:
            print('Reducing perplexity to %d since there are <100 cells in the dataset. ' % perplexity_limit)
        tsne = TSNE(n_components=2, perplexity=perplexity, init='random', random_state=sum(data.shape), n_iter=n_iter,
                    angle=theta)
        key = self.name + ":TSNE:" + "-".join((str(n_components), str(perplexity), str(n_iter), str(theta)))
        self.datadict[key] = pd.DataFrame(tsne.fit_transform(data),
                                 index=self.data.index, columns=['tSNE1', 'tSNE2'])

    def plot_tsne(self, param: str, fig=None, ax=None, density=False, color=None, title='tSNE projection'):
        """Plot tSNE projections of the data
        :param param: 4 parameters of tSNE separated by '-'
        :param fig: matplotlib Figure object
        :param ax: matplotlib Axis object
        :param title: Title for the plot
        """
        fontP = FontProperties()
        fontP.set_size('xx-small')

        tsne_keys = [tsne_key for tsne_key in self.datadict.keys() if 'TSNE' in tsne_key.upper()]
        tsne_match = [key for key in tsne_keys if param in key]

        if not bool(tsne_keys):
            raise RuntimeError('Please run tSNE using run_tsne before plotting ')
        elif not bool(tsne_match):
            par = [int(p) for p in param.split('-')]
            self.run_tsne(par[0], par[1], par[2], par[3])

        tsne = self.datadict[self.name+":TSNE:"+param]

        fig, ax = get_fig(fig=fig, ax=ax)
        if isinstance(color, pd.Series):
            sc = plt.scatter(tsne['tSNE1'], tsne['tSNE2'], s=size,
                             c=color.values, edgecolors='none', cmap='rainbow')
            lp = lambda i: plt.plot([], color=sc.cmap(sc.norm(i)), ms=np.sqrt(size), mec="none",
                                    label="Cluster {:g}".format(i), ls="", marker="o")[0]
            handles = [lp(int(i)) for i in np.unique(color)]
            plt.legend(handles=handles, prop=fontP, loc='upper right').set_frame_on(True)
            # plt.colorbar()
        elif density:
            # Calculate the point density
            xy = np.vstack([tsne['tSNE1'], tsne['tSNE2']])
            z = gaussian_kde(xy)(xy)

            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = tsne['tSNE1'][idx], tsne['tSNE2'][idx], z[idx]

            plt.scatter(x, y, s=size, c=z, edgecolors='none')
            plt.colorbar()
        else:
            plt.scatter(tsne['tSNE1'], tsne['tSNE2'], s=size, edgecolors='none',
                        color=qualitative_colors(2)[1] if color == None else color)

        ax.set_title(title)
        plt.axis('tight')
        plt.tight_layout()
        return fig, ax

    def plot_tsne_by_cell_sizes(self, fig=None, ax=None, vmin=None, vmax=None):
        """Plot tSNE projections of the data with cells colored by molecule counts
        :param fig: matplotlib Figure object
        :param ax: matplotlib Axis object
        :param vmin: Minimum molecule count for plotting
        :param vmax: Maximum molecule count for plotting
        :param title: Title for the plot
        """
        if self.data_type == 'masscyt':
            raise RuntimeError('plot_tsne_by_cell_sizes is not applicable \n\
                for mass cytometry data. ')

        fig, ax = get_fig(fig, ax)
        tsne_keys = [tsne_key for tsne_key in self.datadict.keys() if 'TSNE' in tsne_key.upper()]

        if not bool(tsne_keys):
            raise RuntimeError('Please run run_tsne() before plotting.')
        if self._normalized:
            sizes = self.library_sizes
        else:
            sizes = self.data.sum(axis=1)
        plt.scatter(self.tsne['tSNE1'], self.tsne['tSNE2'], s=size, c=sizes, edgecolors='none')
        plt.colorbar()
        plt.axis('tight')
        plt.tight_layout()
        return fig, ax

    def run_diffusion_map(self, k=10, epsilon=1, distance_metric='euclidean',
                          n_diffusion_components=10, n_pca_components=15, ka=0, random_pca=True):
        """ Run diffusion maps on the data. Run on the principal component projections
        for single cell RNA-seq data and on the expression matrix for mass cytometry data
        :param k: Number of neighbors for graph construction to determine distances between cells
        :param epsilon: Gaussian standard deviation for converting distances to affinities
        :param n_diffusion_components: Number of diffusion components to Generalte
        :param n_pca_components: Number of components to use for running tSNE for single cell
        RNA-seq data. Ignored for mass cytometry
        :return: None
        """

        if not n_pca_components:
            pca_keys = [pca_key for pca_key in self.datadict.keys() if 'PCA' in pca_key.upper()]
            comps = sorted([int(key[key.rfind(':') + 1:]) for key in pca_keys])
            key = self.name + ":PCA:" + str(n_pca_components)
            if n_pca_components in comps:
                data = self.datadict[key]
            else:
                self.run_pca(n_components=n_pca_components, random=random_pca)
                data = self.datadict[key]
        else:
            data = self.data

        N = data.shape[0]

        # Nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, metric=distance_metric).fit(data)
        distances, indices = nbrs.kneighbors(data)

        if ka > 0:
            print('Autotuning distances')
            for j in reversed(range(N)):
                temp = sorted(distances[j])
                lMaxTempIdxs = min(ka, len(temp))
                if lMaxTempIdxs == 0 or temp[lMaxTempIdxs] == 0:
                    distances[j] = 0
                else:
                    distances[j] = np.divide(distances[j], temp[lMaxTempIdxs])

        # Adjacency matrix
        rows = np.zeros(N * k, dtype=np.int32)
        cols = np.zeros(N * k, dtype=np.int32)
        dists = np.zeros(N * k)
        location = 0
        for i in range(N):
            inds = range(location, location + k)
            rows[inds] = indices[i, :]
            cols[inds] = i
            dists[inds] = distances[i, :]
            location += k
        if epsilon > 0:
            W = csr_matrix((dists, (rows, cols)), shape=[N, N])
        else:
            W = csr_matrix((np.ones(dists.shape), (rows, cols)), shape=[N, N])

        # Symmetrize W
        W = W + W.T

        if epsilon > 0:
            # Convert to affinity (with selfloops)
            rows, cols, dists = find(W)
            rows = np.append(rows, range(N))
            cols = np.append(cols, range(N))
            dists = np.append(dists / (epsilon ** 2), np.zeros(N))
            W = csr_matrix((np.exp(-dists), (rows, cols)), shape=[N, N])

        # Create D
        D = np.ravel(W.sum(axis=1))
        D[D != 0] = 1 / D[D != 0]

        # markov normalization
        T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(W)

        # Eigen value decomposition
        D, V = eigs(T, n_diffusion_components, tol=1e-4, maxiter=1000)
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]
        V = T.dot(V)

        # Normalize
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / norm(V[:, i])
        V = np.round(V, 10)

        # Update object
        par = '-'.join((str(k), str(epsilon), str(distance_metric), str(n_diffusion_components), str(n_pca_components),
               str(ka),str(random_pca)))
        df_key = self.name + ":DM:" + par
        diffusion_eigenvectors = pd.DataFrame(V, index=self.data.index,
                                                   columns=['DC' + str(i) for i in range(n_diffusion_components)])
        diffusion_eigenvalues = pd.DataFrame(D)
        self.datadict[df_key] = (diffusion_eigenvectors, diffusion_eigenvalues)

    # def plot_diffusion_components(self, other_data=None, title='Diffusion Components')

    # def plot_diffusion_eigen_vectors(self, fig=None, ax=None, title='Diffusion eigen vectors')

    # def run_diffusion_map_correlations(self, components=None, no_cells=10)

    # def plot_gene_component_correlations(self, components=None, fig=None, ax=None,
    # title='Gene vs. Diffusion Component Correlations')

    # def plot_gene_expression(self, genes, other_data=None)

    def scatter_gene_expression(self, genes, density=False, color=None, fig=None, ax=None):
        """ 2D or 3D scatter plot of expression of selected genes
        :param genes: Iterable of strings to scatter
        """

        not_in_dataframe = set(genes).difference(self.extended_data.columns.get_level_values(1))
        if not_in_dataframe:
            if len(not_in_dataframe) < len(genes):
                print('The following genes were either not observed in the experiment, '
                      'or the wrong gene symbol was used: {!r}'.format(not_in_dataframe))
            else:
                print('None of the listed genes were observed in the experiment, or the '
                      'wrong symbols were used.')
            return

        if len(genes) < 2 or len(genes) > 3:
            raise RuntimeError('Please specify either 2 or 3 genes to scatter.')

        for i in range(len(genes)):
            genes[i] = self.extended_data.columns.values[
                np.where([genes[i] in col for col in self.extended_data.columns.values])[0]][0]

        gui_3d_flag = True
        if ax == None:
            gui_3d_flag = False

        fig, ax = get_fig(fig=fig, ax=ax)
        if len(genes) == 2:
            if density == True:
                # Calculate the point density
                xy = np.vstack([self.extended_data[genes[0]], self.extended_data[genes[1]]])
                z = gaussian_kde(xy)(xy)

                # Sort the points by density, so that the densest points are plotted last
                idx = z.argsort()
                x, y, z = self.extended_data[genes[0]][idx], self.extended_data[genes[1]][idx], z[idx]

                plt.scatter(x, y, s=size, c=z, edgecolors='none')
                ax.set_title('Color = density')
                plt.colorbar()
            elif isinstance(color, pd.Series):
                plt.scatter(self.extended_data[genes[0]], self.extended_data[genes[1]],
                            s=size, c=color, edgecolors='none')
                ax.set_title('Color = ' + color.name)
                plt.colorbar()
            elif color in self.extended_data.columns.get_level_values(1):
                color = self.extended_data.columns.values[
                    np.where([color in col for col in self.extended_data.columns.values])[0]][0]
                plt.scatter(self.extended_data[genes[0]], self.extended_data[genes[1]],
                            s=size, c=self.extended_data[color], edgecolors='none')
                ax.set_title('Color = ' + color[1])
                plt.colorbar()
            else:
                plt.scatter(self.extended_data[genes[0]], self.extended_data[genes[1]], edgecolors='none',
                            s=size, color=qualitative_colors(2)[1] if color == None else color)
            ax.set_xlabel(genes[0][1])
            ax.set_ylabel(genes[1][1])

        else:
            if not gui_3d_flag:
                ax = fig.add_subplot(111, projection='3d')

            if density == True:
                xyz = np.vstack([self.extended_data[genes[0]], self.extended_data[genes[1]],
                                 self.extended_data[genes[2]]])
                kde = gaussian_kde(xyz)
                density = kde(xyz)

                p = ax.scatter(self.extended_data[genes[0]], self.extended_data[genes[1]], self.extended_data[genes[2]],
                               s=size, c=density, edgecolors='none')
                ax.set_title('Color = density')
                fig.colorbar(p)
            elif isinstance(color, pd.Series):
                p = ax.scatter(self.extended_data[genes[0]], self.extended_data[genes[1]],
                               self.extended_data[genes[2]], s=size, c=color, edgecolors='none')
                ax.set_title('Color = ' + color.name)
                fig.colorbar(p)
            elif color in self.extended_data.columns.get_level_values(1):
                color = self.extended_data.columns.values[
                    np.where([color in col for col in self.extended_data.columns.values])[0]][0]
                p = ax.scatter(self.extended_data[genes[0]], self.extended_data[genes[1]],
                               self.extended_data[genes[2]], s=size, c=self.extended_data[color], edgecolors='none')
                ax.set_title('Color = ' + color[1])
                fig.colorbar(p)
            else:
                p = ax.scatter(self.extended_data[genes[0]], self.extended_data[genes[1]], self.extended_data[genes[2]],
                               edgecolors='none', s=size, color=qualitative_colors(2)[1] if color == None else color)
            ax.set_xlabel(genes[0][1])
            ax.set_ylabel(genes[1][1])
            ax.set_zlabel(genes[2][1])
            ax.view_init(15, 55)

        plt.axis('tight')
        plt.tight_layout()
        return fig, ax

    # def scatter_gene_expression_against_other_data(self, genes, other_data, density=False, color=None, fig=None,
    #                                               ax=None)

    def run_magic(self, n_pca_components=20, random_pca=True, t=6, k=30, ka=10, epsilon=1, rescale_percent=99):
        par = '-'.join(str(n_pca_components), str(random_pca), str(t), str(k), str(ka), str(epsilon), str(rescale_percent))
        key = self.name + ":MAGIC:" + par
        new_data = magic.MAGIC.magic(self.data.values, n_pca_components=n_pca_components, random_pca=random_pca, t=t,
                                     k=k, ka=ka, epsilon=epsilon, rescale=rescale_percent)

        new_data = pd.DataFrame(new_data, index=self.data.index, columns=self.data.columns)

        # Construct class object
        scdata = magic.mg.SCData(key, new_data, data_type=self.data_type)
        self.datadict[key] = scdata

    def concatenate_data(self, other_data_sets, join='outer', axis=0, names=[]):

        # concatenate dataframes
        temp = self.data.copy()
        if axis == 0:
            temp.index = [str(names[0]) + ' ' + str(i) for i in self.data.index]
        else:
            temp.columns = [str(names[0]) + ' ' + str(i) for i in self.data.columns]
        dfs = [temp]
        count = 0
        for data_set in other_data_sets:
            count += 1
            temp = data_set.data.copy()
            if axis == 0:
                temp.index = [str(names[count]) + ' ' + str(i) for i in data_set.data.index]
            else:
                temp.columns = [str(names[count]) + ' ' + str(i) for i in self.data.columns]
            dfs.append(temp)
        df_concat = pd.concat(dfs, join=join, axis=axis)

        scdata = magic.mg.SCData(self.name + "concatenated", df_concat)
        return scdata


class ClusterInfo:
    def __init__(self, communities, graph, Q, source, method='phenograph'):
        if not isinstance(communities, np.ndarray):
            raise TypeError("communities must be a numpy array")
        elif not isinstance(graph, coo.coo_matrix):
            raise TypeError("graph must be of type scipy.coo.coo_matrix")
        elif not isinstance(Q, float):
            raise TypeError("modularity score Q must be float")
        self._cluster = communities
        self._graph = graph
        self._modscore = Q

    @property
    def cluster(self):
        return self._cluster

    @property
    def graph(self):
        return self._graph

    @property
    def modscore(self):
        return self._modscore

