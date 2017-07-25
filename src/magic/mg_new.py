import re
import os
import random
import pickle
import shlex
import shutil
from copy import deepcopy
from collections import defaultdict, Counter
from subprocess import call, Popen, PIPE
import glob
import warnings
import unittest

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
import sys

sys.path.insert(0, '/Users/vincentliu/PycharmProjects/magic/src/magic')
import MAGIC
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


class ClusterInfo:
    def __init__(self, communities, graph, Q, method='phenograph'):
        if not isinstance(communities, np.ndarray):
            raise TypeError("communities must be a numpy array")
        elif not isinstance(graph, coo.coo_matrix):
            raise TypeError("graph must be of type scipy.coo.coo_matrix")
        elif not isinstance(Q, float):
            raise TypeError("modularity score Q must be float")
        self._cluster = communities
        self._graph = graph
        self._modscore = Q
        self._method = method

    @property
    def cluster(self):
        return self._cluster

    @property
    def graph(self):
        return self._graph

    @property
    def modscore(self):
        return self._modscore

    @property
    def method(self):
        return self._method


class Operations:
    def __init__(self, sourcename: str = None, inherite: list = None):
        if not sourcename and not inherite:
            raise RuntimeError("sourcename and inherite can't both be None")
        self._history = deepcopy(inherite) if inherite else [sourcename]
        self._operations = ("PCA", "DM", "MAGIC", "PHENOGRAPH", "LOGTRANS", "NORMALIZED", "FILTERED")

    @property
    def history(self):
        return self._history

    def add(self, op: str, params: str = ''):
        if op not in self._operations:
            raise RuntimeError("Invalid operation.")
        cur_op = op + ':' + params if op not in ['NORMALIZED', 'LOGTRANS', "FILTERED"] else op
        self._history.append(cur_op)

    def clear(self):
        self._history.clear()


class SCData:
    def __init__(self, name: str, data, data_type='sc-seq', metadata=None, operation: Operations = None):
        if not (isinstance(data, pd.DataFrame)):
            raise TypeError('data must be of type DataFrame')
        if data_type not in ['sc-seq', 'masscyt']:
            raise RuntimeError('data type must be either sc-seq or masscyt')
        if metadata is None:
            metadata = pd.DataFrame(index=data.index, dtype='O')

        # initiate the data dictionary with the given data
        self._name = name
        cols = [np.array(data.columns.values)]
        self._datadict = {'original ' + name: pd.DataFrame(data.values, index=data.index, columns=cols)}
        self._data_type = data_type
        self._metadata = metadata

        self._operation = Operations(sourcename=self.name) if operation is None \
            else Operations(inherite=operation.history)

        self._clusterinfo = None

        # Library size
        self._library_sizes = None

    def reset(self):
        self._datadict.clear()
        self._library_sizes = None

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
        c, g = self.data.shape
        _repr = ('SCData: {c} cells x {g} genes\n'.format(g=g, c=c))
        _repr += '\n{} = {}'.format('name', self._name)
        _repr += '\n{} = {}'.format('data type', self._data_type)
        _repr += '\n{} = {}'.format('number of derived datasets', len(self._datadict) - 1)
        _repr += '\n{}: {}'.format('operation history', ' '.join(self._operation.history))
        return _repr

    @property
    def name(self):
        return self._name

    @property
    def data_type(self):
        return self._data_type

    @property
    def operation(self):
        return self._operation

    # returns the data dictionary
    @property
    def datadict(self):
        return self._datadict

    # returns the raw data
    @property
    def data(self):
        return self._datadict['original ' + self.name]

    @data.setter
    def data(self, item):
        if not (isinstance(item, pd.DataFrame)):
            raise TypeError('SCData.data must be of type DataFrame')
        self.reset()
        cols = [np.array(item.columns.values)]
        self._datadict = {'original ' + self.name: pd.DataFrame(item.values, index=item.index, columns=cols)}

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, item):
        if not isinstance(item, pd.DataFrame):
            raise TypeError('SCData.metadata must be of type DataFrame')
        self._metadata = item

    @property
    def cluster(self):
        return self._clusterinfo

    @cluster.setter
    def cluster(self, cluobj):
        if type(cluobj) is not type(ClusterInfo):
            raise TypeError("cluster must be a ClusterInfo object")
        self._clusterinfo = cluobj

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
                 rows_after_header_to_skip=0, cols_after_header_to_skip=0):

        if data_type not in ['sc-seq', 'masscyt']:
            raise RuntimeError('data_type must be either sc-seq or masscyt')

        # Read in csv file
        df = pd.DataFrame.from_csv(counts_csv_file, sep=delimiter)

        df.drop(df.index[1:rows_after_header_to_skip + 1], axis=0, inplace=True)
        df.drop(df.columns[1:cols_after_header_to_skip + 1], axis=1, inplace=True)

        if cell_axis != 0:
            df = df.transpose()

        # Construct class object
        scdata = cls(data_name, df, data_type=data_type)

        return scdata

    @classmethod
    def from_fcs(cls, fcs_file, data_name: str, cofactor=5,
                 metadata_channels=('Time', 'Event_length', 'DNA1', 'DNA2', 'Cisplatin', 'beadDist', 'bead1')):

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

    def normalize_scseq_data(self):
        """
        Normalize single cell RNA-seq data: Divide each cell by its molecule count
        and multiply counts of cells by the median of the molecule counts
        This is the only operation that changes data directly rather than adding a new SCData object
        """
        if 'NORMALIZED' in self.operation.history:
            pass
        else:
            molecule_counts = self.data.sum(axis=1)
            self.data = self.data.div(molecule_counts, axis=0) \
                .mul(np.median(molecule_counts), axis=0)

            # check that none of the genes are empty; if so remove them
            nonzero_genes = self.data.sum(axis=0) != 0
            self.data = self.data.loc[:, nonzero_genes].astype(np.float32)

            # set unnormalized_cell_sums
            self.library_sizes = molecule_counts
            self.operation.add('NORMALIZED')

    def filter_scseq_data(self, filter_cell_min=0, filter_cell_max=np.inf, filter_gene_nonzero=None,
                          filter_gene_mols=None):

        if len(self.operation.history) != 1:
            print("data must be filtered before any other operation is performed")

        else:
            sums = self.data.sum(axis=1)
            to_keep = np.intersect1d(np.where(sums >= filter_cell_min)[0],
                                     np.where(sums <= filter_cell_max)[0])
            self.data = self.data.loc[self.data.index[to_keep], :].astype(np.float32)

            if filter_gene_nonzero is not None:
                nonzero = self.data.astype(bool).sum(axis=0)
                to_keep = np.where(nonzero >= filter_gene_nonzero)[0]
                self.data = self.data.loc[[]:, to_keep].astype(np.float32)

            if filter_gene_mols is not None:
                sums = self.data.sum(axis=0)
                to_keep = np.where(sums >= filter_gene_mols)[0]
                self.data = self.data.loc[:, to_keep].astype(np.float32)

            self.operation.add("FILTERED")

    def log_transform_scseq_data(self):
        if 'LOGTRANS' in self.operation.history:
            return
        else:
            self.data = np.log(np.add(self.data, 0.1))
            self.operation.add('LOGTRANS')

    def run_magic(self, n_pca_components=20, random_pca=True, t=6, k=30, ka=10, epsilon=1, rescale_percent=99):

        MAGICed = [x for x in self.operation.history if 'MAGIC' in x]

        if MAGICed:
            print('already ran MAGIC on the current data')
            return

        else:
            pca_data = self.run_pca(n_pca_components, random_pca, no_effect=True)

            new_data = MAGIC.magic(self.data.values, pca_data.data.values,
                                   t=t, k=k, ka=ka, epsilon=epsilon, rescale=rescale_percent)
            new_data = pd.DataFrame(new_data, index=self.data.index, columns=self.data.columns)

            # Construct class object
            par = '-'.join((str(n_pca_components), str(random_pca), str(t), str(k),
                            str(ka), str(epsilon), str(rescale_percent)))
            self.data = new_data
            self.operation.add('MAGIC', par)

    def run_pca(self, n_components=100, rand=True, no_effect=False):
        """
        Principal component analysis of the data.
        Note: Column values for the old method are (dataname, PCX) now its just PCX
        name: source data name:PCA:parameters joined by -:number
        :param n_components: Number of components to project the data
        :param rand: Whether randomized
        """
        solver = 'randomized' if rand else 'full'

        pca = PCA(n_components=n_components, svd_solver=solver)
        new_data = pd.DataFrame(data=pca.fit_transform(self.data.values), index=self.data.index,
                                columns=['PC' + str(i) for i in range(1, n_components + 1)])

        # assuming each run is different even with the same parameters
        key_base = self.operation.history[0] + ":PCA:" + str(n_components)
        count = 0
        key = key_base + ':' + str(count)
        while key in self.datadict.keys():
            count += 1
            key = key_base + ':' + str(count)

        scdata = SCData(key, new_data, self.data_type, self.metadata, self.operation)
        scdata.operation.add('PCA', str(n_components))

        if not no_effect:
            self.datadict[key] = scdata
        else:
            pass

        return scdata

    def run_diffusion_map(self, k=10, epsilon=1, distance_metric='euclidean',
                          n_diffusion_components=10, ka=0):
        """ Run diffusion maps on the data. Run on the principal component projections
        for single cell RNA-seq data and on the expression matrix for mass cytometry data
        :param k: Number of neighbors for graph construction to determine distances between cells
        :param epsilon: Gaussian standard deviation for converting distances to affinities
        :param n_diffusion_components: Number of diffusion components to Generalte
        :return: None
        """
        #  if self.data_type == 'sc-seq' and 'PCA' not in self.operation.history[-1]:
        #     print("must provide pcadata for scRNA sequencing data")
        #     return

        N = self.data.shape[0]

        # Nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, metric=distance_metric).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)

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
        diffusion_eigenvectors = pd.DataFrame(V, index=self.data.index,
                                              columns=['DC' + str(i) for i in range(n_diffusion_components)])
        diffusion_eigenvalues = pd.DataFrame(D)

        # assuming each run is different even with the same parameters
        par = '-'.join((str(k), str(epsilon), str(distance_metric), str(n_diffusion_components), str(ka)))
        key_base = self.operation.history[0] + ":DM:" + par
        count = 0
        key = key_base + ':' + str(count)
        while key in self.datadict.keys():
            count += 1
            key = key_base + ':' + str(count)

        scdata = SCData(key, diffusion_eigenvectors, self.data_type, self.metadata, self.operation)
        scdata.operation.add('DM', par)
        self.datadict[key] = scdata

        return scdata

    def run_phenograph(self, k=30, directed=False, prune=False, min_cluster_size=10, jaccard=True,
                       dis_metric='euclidean', n_jobs=-1, q_tol=1e-3, louvain_time_limit=2000, nn_method='kdtree'):
        communities, graph, Q = phenograph.cluster(self.data, k=k,
                                                   directed=directed, prune=prune,
                                                   min_cluster_size=min_cluster_size, jaccard=jaccard,
                                                   primary_metric=dis_metric, n_jobs=n_jobs,
                                                   q_tol=q_tol, louvain_time_limit=louvain_time_limit,
                                                   nn_method=nn_method)
        self.cluster = ClusterInfo(communities, graph, Q, 'phenograph')

        return communities

    # note now this only returns a pd.Dataframe object
    def run_tsne(self, perplexity=30, n_iter=1000, theta=0.5):
        """ Run tSNE on the data. tSNE is run on the principal component projections
        for single cell RNA-seq data and on the expression matrix for mass cytometry data
        stored in the data dictionary as "name:TSNE:comp-perp-iter-theta"
        Note: Column values for the old method are (dataname, tSNE) now its just tSNE
        :param n_components: Number of components to use for running tSNE for single cell
        RNA-seq data. Ignored for mass cytometry
        :return: None
        """
        og_name = self.name if self.name.find(':') == -1 else self.name[:self.name.find(':')]

        # Reduce perplexity if necessary
        perplexity_limit = 15

        if self.data.shape[0] < 100 and perplexity > perplexity_limit:
            print('Reducing perplexity to %d since there are <100 cells in the dataset. ' % perplexity_limit)
            perp = perplexity_limit
        else:
            perp = perplexity
        tsne = TSNE(n_components=2, perplexity=perp, init='random', random_state=sum(self.data.shape),
                    n_iter=n_iter, angle=float(theta))

        tsne_data = pd.DataFrame(tsne.fit_transform(self.data), index=self.data.index, columns=['tSNE1', 'tSNE2'])

        return tsne_data

    def plot_molecules_per_cell_and_gene(self, fig=None, ax=None):
        if len(self.operation.history) != 1:
            print("plotting molecules per cell and gene is only possible on unprocessed data")
        else:
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

    @staticmethod
    def plot_tsne(tsne, fig=None, ax=None, density=False, color=None, title='tSNE projection'):
        """Plot tSNE projections of the data
        Must make sure the object being operated contains tSNE data
        :param tsne: pd.Dataframe that contains tsne data
        :param fig: matplotlib Figure object
        :param ax: matplotlib Axis object
        :param title: Title for the plot
        """
        fontP = FontProperties()
        fontP.set_size('xx-small')

        fig, ax = get_fig(fig=fig, ax=ax)
        if isinstance(color, pd.Series):
            sc = plt.scatter(tsne['tSNE1'], tsne['tSNE2'], s=size,
                             c=color.values, edgecolors='none', cmap='rainbow')
            lp = lambda i: plt.plot([], color=sc.cmap(sc.norm(i)), ms=np.sqrt(size), mec="none",
                                    label="Cluster {:g}".format(i), ls="", marker="o")[0]
            handles = [lp(int(i)) for i in np.unique(color)]
            plt.legend(handles=handles, prop=fontP, loc='upper right').set_frame_on(True)
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
                        color=qualitative_colors(2)[1] if color is None else color)

        ax.set_title(title)
        plt.axis('tight')
        plt.tight_layout()
        return fig, ax

    def plot_phenograph(self, communities, fig=None, ax=None, density=False, title='PhenoGraph Clustering'):
        tsne = self.run_tsne()
        self.plot_tsne(tsne, fig=fig, ax=ax, density=density, color=communities, title=title)

        return fig, ax

    def scatter_gene_expression(self, genes, density=False, color=None, fig=None, ax=None):
        """ 2D or 3D scatter plot of expression of selected genes
        :param genes: Iterable of strings to scatter
        """

        not_in_dataframe = set(genes).difference(self.data.columns)

        print(genes)
        if not_in_dataframe:
            if len(not_in_dataframe) < len(genes):
                print('The following genes were either not observed in the experiment, '
                      'or the wrong gene symbol was used: {!r}'.format(not_in_dataframe))
            else:
                print('None of the listed genes were observed in the experiment, or the '
                      'wrong symbols were used.')
            return

        if len(genes) not in [2, 3]:
            raise RuntimeError('Please specify either 2 or 3 genes to scatter.')

        gui_3d_flag = False if ax is None else True

        x, y = self.data[genes[0]].to_frame(), self.data[genes[1]].to_frame()

        fig, ax = get_fig(fig=fig, ax=ax)
        if len(genes) == 2:
            if density is True:
                # Calculate the point density
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy)

                # Sort the points by density, so that the densest points are plotted last
                idx = z.argsort()
                x, y, z = x[idx], y[idx], z[idx]

                plt.scatter(x, y, s=size, c=z, edgecolors='none')
                ax.set_title('Color = density')
                plt.colorbar()

            elif isinstance(color, pd.Series):
                plt.scatter(x, y, s=size, c=color, edgecolors='none')
                ax.set_title('Color = ' + color.name)
                plt.colorbar()

            # elif color in self.extended_data.columns.get_level_values(1):
            #     color = self.extended_data.columns.values[
            #         np.where([color in col for col in self.extended_data.columns.values])[0]][0]
            #     plt.scatter(x, y, s=size, c=self.extended_data[color], edgecolors='none')
            #     ax.set_title('Color = ' + color[1])
            #     plt.colorbar()

            else:
                plt.scatter(x, y, edgecolors='none',
                            s=size, color=qualitative_colors(2)[1] if color is None else color)
            ax.set_xlabel(genes[0][1])
            ax.set_ylabel(genes[1][1])

        else:
            z = self.data[genes[2]].to_frame()

            if not gui_3d_flag:
                ax = fig.add_subplot(111, projection='3d')

            if density is True:
                xyz = np.vstack([x, y, z])
                kde = gaussian_kde(xyz)
                density = kde(xyz)

                p = ax.scatter(x, y, z, s=size, c=density, edgecolors='none')
                ax.set_title('Color = density')
                fig.colorbar(p)

            elif isinstance(color, pd.Series):
                p = ax.scatter(x, y, z, s=size, c=color, edgecolors='none')
                ax.set_title('Color = ' + color.name)
                fig.colorbar(p)

            # elif color in self.extended_data.columns.get_level_values(1):
            #    color = self.extended_data.columns.values[
            #         np.where([color in col for col in self.extended_data.columns.values])[0]][0]
            #     p = ax.scatter(x, y, z, s=size, c=self.extended_data[color], edgecolors='none')
            #     ax.set_title('Color = ' + color[1])
            #     fig.colorbar(p)

            else:
                p = ax.scatter(x, y, z,
                               edgecolors='none', s=size, color=qualitative_colors(2)[1] if color is None else color)
                ax.set_title('Color = ')
                # fig.colorbar(p)

            ax.set_xlabel(genes[0][1])
            ax.set_ylabel(genes[1][1])
            ax.set_zlabel(genes[2][1])
            ax.view_init(15, 55)

        plt.axis('tight')
        plt.tight_layout()
        return fig, ax

    def concatenate_data(self, other_data_sets, names=(), join='outer', axis=0):

        # concatenate dataframes
        temp = deepcopy(self.data)
        if axis == 0:
            temp.index = [str(names[0]) + ' ' + str(i) for i in self.data.index]
        else:
            temp.columns = [str(names[0]) + ' ' + str(i) for i in self.data.columns]

        self.datadict.clear()  # delete all derived SCData objects
        self.operation.clear()

        dfs = [temp]
        count = 0

        for data_set in other_data_sets:
            count += 1
            temp = data_set.data.copy()
            if axis == 0:
                temp.index = [str(names[count]) + ' ' + str(i) for i in data_set.data.index]
            else:
                temp.columns = [str(names[count]) + ' ' + str(i) for i in data_set.data.columns]
            dfs.append(temp)
            data_set.datadict.clear()
            data_set.operation.clear()

        df_concat = pd.concat(dfs, join=join, axis=axis)

        scdata = SCData(self.name + " concatenated", df_concat, self.data_type)

        return scdata

    @classmethod
    def retrieve_data(cls, original, opseq: list):
        curscdata = original
        for op in opseq[1:]:
            try:
                curscdata = curscdata.datadict[op]
            except KeyError:
                print(op)
                print(curscdata.name)
                print(curscdata.datadict.keys())

        return curscdata