#!/usr/local/bin/python3

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import reduce, partial
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D graph
import os
import platform
import pandas as pd
import tkinter as tk
import numpy as np
from tkinter import filedialog, ttk
import csv
import scras


class SCRASGui(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self._parent = parent

        self.menubar = tk.Menu(self)
        self.fileMenu = tk.Menu(self.menubar, tearoff=0)
        self.analysisMenu = tk.Menu(self.menubar, tearoff=0)
        self.visMenu = tk.Menu(self.menubar, tearoff=0)

        self.currentPlot = None
        self.data = {}

        self.initialize()

    # updated
    def initialize(self):
        self.grid()

        # set up menu bar
        self.menubar.add_cascade(label="File", menu=self.fileMenu)
        self.fileMenu.add_command(label="Load csv file", command=self.load_csv)
        self.fileMenu.add_command(label="Load sparse data file", command=self.load_mtx)
        self.fileMenu.add_command(label="Load 10x file", command=self.load_10x)
        self.fileMenu.add_command(label="Load saved session from pickle file", command=self.load_pickle)
        self.fileMenu.add_command(label="Concatenate datasets", state='disabled', command=self.concatenate_data)
        self.fileMenu.add_command(label="Save data", state='disabled', command=self.save_data)
        self.fileMenu.add_command(label="Save plot", state='disabled', command=self.save_plot)
        self.fileMenu.add_command(label="Exit", command=self.quit_scras())

        self.menubar.add_cascade(label="Analysis", menu=self.analysisMenu)
        self.analysisMenu.add_command(label="Dimensionality Reduction", state='disabled', command=self.run_dr)
        self.analysisMenu.add_command(label="Clustering", state='disabled', command=self.run_clustering)
        self.analysisMenu.add_command(label="Gene Expression Analysis", state='disabled', command=self.run_gea)

        self.menubar.add_cascade(label="Visualization", menu=self.visMenu)
        self.visMenu.add_command(label="tSNE", state='disabled', command=self.tsne)
        self.visMenu.add_command(label="Scatter plot", state='disabled', command=self.scatter_plot)
        self.visMenu.add_command(label="Gene expression", state='disabled', command=self.gene_expression)

        self.config(menu=self.menubar)

        # intro screen
        tk.Label(self, text="SCRAS", font=('Helvetica', 48), fg="black", bg="white",
                 padx=100, pady=10).grid(row=0)
        tk.Label(self, text="Single Cell RNA Analysis Suite", font=('Helvetica', 25), fg="black",
                 bg="white", padx=100, pady=5).grid(row=1)
        tk.Label(self, text="To get started, select a data file by clicking File > Load Data", fg="black", bg="white",
                 padx=100, pady=25).grid(row=2)

        # update
        self.protocol('WM_DELETE_WINDOW', self.quit_scras())
        self.grid_columnconfigure(0, weight=1)
        self.resizable(False, False)
        self.update()
        self.geometry(self.geometry())
        self.focus_force()

    def load_csv(self):
        self.filename = filedialog.askopenfilename(title='Load data file', initialdir='~/.magic/data')

        if self.filename:
            self.import_options = tk.Toplevel()
            self.import_options.resizable(False, False)
            self.import_options.title('Import options')

            fileNameContainer = tk.Frame(self.import_options)
            fileNameContainer.grid(column=0, row=0, sticky='w')
            tk.Label(fileNameContainer, text="File name: ", pady=5).grid(column=0, row=0, sticky='w')
            tk.Label(fileNameContainer, text=self.filename.split('/')[-1]).grid(column=1, row=0, sticky='w')

            nameEntryContainer = tk.Frame(self.import_options)
            nameEntryContainer.grid(column=0, row=1, sticky='w')
            tk.Label(nameEntryContainer, text="Data name: ").grid(column=0, row=0, sticky='w')
            self.fileNameEntryVar = tk.StringVar()
            self.fileNameEntryVar.set('Data ' + str(len(self.data) + 1))
            tk.Entry(nameEntryContainer, textvariable=self.fileNameEntryVar).grid(column=1, row=0, sticky='w')

            delimiterContainer = tk.Frame(self.import_options)
            delimiterContainer.grid(column=0, row=2, sticky='w')
            tk.Label(delimiterContainer, text="Delimiter: ").grid(column=0, row=0, sticky='w')
            self.delimiter = tk.StringVar()
            self.delimiter.set(',')
            tk.Entry(delimiterContainer, textvariable=self.delimiter).grid(column=1, row=0, sticky='w')

            rowSelectionContainer = tk.Frame(self.import_options)
            rowSelectionContainer.grid(column=0, row=3, sticky='w')
            tk.Label(rowSelectionContainer, text="Rows:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.rowVar = tk.IntVar()
            self.rowVar.set(0)
            tk.Radiobutton(rowSelectionContainer, text="Cells",
                           variable=self.rowVar, value=0).grid(column=1, row=0, sticky='W')
            tk.Radiobutton(rowSelectionContainer, text="Genes",
                           variable=self.rowVar, value=1).grid(column=2, row=0, sticky='W')

            skipSelectionContainer = tk.Frame(self.import_options)
            skipSelectionContainer.grid(column=0, row=4, sticky='w')
            tk.Label(skipSelectionContainer,
                     text="Number of additional rows/columns to skip after gene/cell names").grid(
                column=0, row=0, columnspan=3, sticky='W')

            numRowContainer = tk.Frame(skipSelectionContainer)
            numRowContainer.grid(column=0, row=1, sticky='W')
            tk.Label(numRowContainer, text="Number of rows:").grid(column=0, row=0, sticky='W')
            self.rowHeader = tk.IntVar()
            self.rowHeader.set(0)
            tk.Entry(numRowContainer, textvariable=self.rowHeader).grid(column=1, row=0, sticky='W')

            numColContainer = tk.Frame(skipSelectionContainer)
            numColContainer.grid(column=0, row=2, sticky='W')
            tk.Label(numColContainer, text="Number of columns:").grid(column=0, row=0, sticky='W')
            self.colHeader = tk.IntVar()
            self.colHeader.set(0)
            tk.Entry(numColContainer, textvariable=self.colHeader).grid(column=1, row=0, sticky='W')

            tk.Button(self.import_options, text="Compute data statistics",
                      command=partial(self.showRawDataDistributions, file_type='csv')).grid(column=0, row=5, padx=8)

            ttk.Separator(self.import_options, orient='horizontal').grid(column=0, row=6, sticky='ew', pady=8)

            tk.Label(self.import_options, text="Filtering options (leave blank if no filtering)",
                     fg="black", bg="white", font="bold").grid(column=0, row=7, sticky='w')

            # filter parameters
            molPerCellContainer = tk.Frame(self.import_options)
            molPerCellContainer.grid(column=0, row=8, sticky='w')
            self.filterCellMinVar = tk.StringVar()
            tk.Label(molPerCellContainer, text="Filter by molecules per cell  Min:",
                     fg="black", bg="white").grid(column=0, row=0, sticky='w')
            tk.Entry(molPerCellContainer, textvariable=self.filterCellMinVar).grid(column=1, row=0, sticky='w')
            self.filterCellMaxVar = tk.StringVar()
            tk.Label(molPerCellContainer, text="Max:",
                     fg="black", bg="white").grid(column=0, row=1, sticky='E')
            tk.Entry(molPerCellContainer, textvariable=self.filterCellMaxVar).grid(column=1, row=1, sticky='w')

            cellPerGeneContainer = tk.Frame(self.import_options)
            cellPerGeneContainer.grid(column=0, row=9, sticky='w')
            self.filterGeneNonzeroVar = tk.StringVar()
            tk.Label(cellPerGeneContainer, text="Filter by nonzero cells per gene  Min:", fg="black", bg="white").grid(
                column=0, row=0, sticky='w')
            tk.Entry(cellPerGeneContainer, textvariable=self.filterGeneNonzeroVar).grid(column=1, row=0, sticky='w')

            molPerGeneContainer = tk.Frame(self.import_options)
            molPerGeneContainer.grid(column=0, row=10, sticky='w')
            self.filterGeneMolsVar = tk.StringVar()
            tk.Label(molPerGeneContainer, text="Filter by molecules per gene. Min:",
                     fg="black", bg="white").grid(column=0, row=0, sticky='w')
            tk.Entry(molPerGeneContainer, textvariable=self.filterGeneMolsVar).grid(column=1, row=0, sticky='w')

            # horizontal separator
            ttk.Separator(self.import_options, orient='horizontal').grid(column=0, row=11, sticky='ew', pady=8)

            tk.Label(self.import_options, text="Data pre-processing options",
                     fg="black", bg="white").grid(column=0, row=12, sticky='w')

            checkButtonContainer = tk.Frame(self.import_options)
            checkButtonContainer.grid(column=0, row=13, sticky='w')
            # normalize
            self.normalizeVar = tk.BooleanVar()
            self.normalizeVar.set(True)
            tk.Checkbutton(checkButtonContainer, text="Normalize by library size",
                           variable=self.normalizeVar).grid(column=0, row=0, sticky='w')
            # log transform
            self.logTransform = tk.BooleanVar()
            self.logTransform.set(True)
            tk.Checkbutton(checkButtonContainer,
                           text="Log-transform data", variable=self.logTransform,
                           command=self._update_mg_options).grid(column=1, row=0, sticky='w')
            # MAGIC
            self.magicVar = tk.BooleanVar()
            self.magicVar.set(False)
            tk.Checkbutton(checkButtonContainer,
                           text="Run MAGIC on the data", variable=self.magicVar,
                           command=self._update_mg_options).grid(column=3, row=0, sticky='w')

            # MAGIC options
            tk.Label(self.import_options, text="MAGIC options").grid(column=0, row=14, pady=8, sticky='w')

            mgPCACompContainer = tk.Frame(self.import_options)
            mgPCACompContainer.grid(column=0, row=15, sticky='w')
            tk.Label(mgPCACompContainer, text="Number of PCA components:", fg="black", bg="white").grid(column=0, row=0)
            self.mgCompVar = tk.IntVar()
            self.mgCompVar.set(20)
            self.mgPCAEntry = tk.Entry(mgPCACompContainer, textvariable=self.mgCompVar, state='disabled')
            self.mgPCAEntry.grid(column=1, row=0)
            self.mgRandomVar = tk.BooleanVar()
            self.mgRandomVar.set(True)
            self.mgRandCheckButton = tk.Checkbutton(mgPCACompContainer, text="Randomized PCA",
                                                    variable=self.mgRandomVar, state='disabled')
            self.mgRandCheckButton.grid(column=1, row=1, sticky='W')

            mgTContainer = tk.Frame(self.import_options)
            mgTContainer.grid(column=0, row=16, sticky='W')
            tk.Label(mgTContainer, text="t:", fg="black", bg="white").grid(column=0, row=0, sticky='W')
            self.mgTVar = tk.IntVar()
            self.mgTVar.set(6)
            self.mgTVarEntry = tk.Entry(mgTContainer, textvariable=self.mgTVar, state='disabled')
            self.mgTVarEntry.grid(column=1, row=0, sticky='W')

            mgKContainer = tk.Frame(self.import_options)
            mgKContainer.grid(column=0, row=17, sticky='w')
            tk.Label(mgKContainer, text="k:", fg="black", bg="white").grid(column=0, row=0, sticky='W')
            self.mgKVar = tk.IntVar()
            self.mgKVar.set(30)
            self.mgKVarEntry = tk.Entry(mgKContainer, textvariable=self.mgKVar, state='disabled')
            self.mgKVarEntry.grid(column=1, row=0, sticky='W')

            mgKaContainer = tk.Frame(self.import_options)
            mgKaContainer.grid(column=0, row=18, sticky='w')
            tk.Label(mgKaContainer, text="ka:", fg="black", bg="white").grid(column=0, row=0, sticky='W')
            self.mgKaVar = tk.IntVar()
            self.mgKaVar.set(10)
            self.mgKaVarEntry = tk.Entry(mgKaContainer, textvariable=self.mgKaVar, state='disabled')
            self.mgKaVarEntry.grid(column=1, row=0, sticky='W')

            mgEpContainer = tk.Frame(self.import_options)
            mgEpContainer.grid(column=0, row=19, sticky='w')
            tk.Label(mgEpContainer, text="Epsilon:", fg="black", bg="white").grid(column=0, row=0, sticky='W')
            self.mgEpVar = tk.IntVar()
            self.mgEpVar.set(1)
            self.mgEPVar = tk.Entry(mgEpContainer, textvariable=self.mgEpVar, state='disabled')
            self.mgEPVar.grid(column=1, row=0, sticky='W')
            tk.Label(mgEpContainer, text="(0 is the uniform kernel)",
                     fg="black", bg="white").grid(column=2, row=0)

            mgRescaleContainer = tk.Frame(self.import_options)
            mgRescaleContainer.grid(column=0, row=20, sticky='w')
            self.mgRescaleVar = tk.IntVar()
            self.mgRescaleVar.set(99)
            tk.Label(mgRescaleContainer, text="Rescale data to ",
                     fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.mgResEntry = tk.Entry(mgRescaleContainer, textvariable=self.mgRescaleVar, state='disabled')
            self.mgResEntry.grid(column=1, row=0, sticky='w')
            tk.Label(mgRescaleContainer, text=" percentile (use 0 for log-transformed data)",
                     fg="black", bg="white").grid(column=2, row=0, sticky='w')

            finalButtonContainer = tk.Frame(self.import_options)
            finalButtonContainer.grid(column=0, row=21)
            tk.Button(finalButtonContainer, text="Cancel", command=self.import_options.destroy).grid(column=0, row=0,
                                                                                                     padx=20)
            tk.Button(finalButtonContainer, text="Load",
                      command=partial(self.process_data, file_type='csv')).grid(column=1, row=0, padx=20)

            self.wait_window(self.import_options)

    def load_mtx(self):
        pass  # to be implemented

    def load_10x(self):
        pass  # to be implemented

    def load_pickle(self):
        pass  # to be implemented

    def _update_mg_options(self):
        if self.magicVar.get():
            self.mgPCAEntry.config(state='normal')
            self.mgRandCheckButton.config(state='normal')
            self.mgTVarEntry.config(state='normal')
            self.mgKVarEntry.config(state='normal')
            self.mgKaVarEntry.config(state='normal')
            self.mgEPVar.config(state='normal')
            if self.logTransform.get():
                self.mgRescaleVar.set(0)
                self.mgResEntry.config(state='disabled')
            else:
                self.mgRescaleVar.set(99)
                self.mgResEntry.config(state='normal')
        else:
            self.mgPCAEntry.config(state='disabled')
            self.mgRandCheckButton.config(state='disabled')
            self.mgTVarEntry.config(state='disabled')
            self.mgKVarEntry.config(state='disabled')
            self.mgKaVarEntry.config(state='disabled')
            self.mgEPVar.config(state='disabled')
            self.mgResEntry.config(state='disabled')

    def showRawDataDistributions(self, file_type='csv'):
        if file_type == 'csv':  # sc-seq data
            scdata = scras.SCData.from_csv(os.path.expanduser(self.filename), data_name='preprocess', data_type='sc-seq')
        elif file_type == 'mtx':  # sparse matrix
            scdata = scras.SCData.from_mtx(os.path.expanduser(self.dataFileName),
                                        os.path.expanduser(self.geneNameFile))
        elif file_type == '10x':
            scdata = scras.SCData.from_10x(self.dataDir)

        self.dataDistributions = tk.Toplevel()
        self.dataDistributions.title(self.fileNameEntryVar.get() + ": raw data distributions")

        fig, ax = scdata.plot_molecules_per_cell_and_gene()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.dataDistributions)
        canvas.show()
        canvas.get_tk_widget().grid(column=0, row=0, rowspan=10, columnspan=4, sticky='NSEW')
        del scdata

        self.wait_window(self.dataDistributions)

    def process_data(self, file_type='csv'):

        if len(self.data) == 0:
            # clear intro screen
            for item in self.grid_slaves():
                item.grid_forget()

            # list for datasets
            self.data_list = ttk.Treeview()
            self.data_list.heading('#0', text='Data sets')
            self.data_list.grid(column=0, row=0, rowspan=6, sticky='NSEW')
            self.data_list.bind('<BackSpace>', self._deleteDataItem)
            self.data_list.bind('<<TreeviewSelect>>', self._updateSelection)
            # make Treeview scrollable
            ysb = ttk.Scrollbar(orient=tk.VERTICAL, command=self.data_list.yview)
            xsb = ttk.Scrollbar(orient=tk.HORIZONTAL, command=self.data_list.xview)
            self.data_list.configure(yscroll=ysb.set, xscroll=xsb.set)

            # list for features of the selected dataset
            self.data_detail = ttk.Treeview()
            self.data_detail.heading('#0', text='Features')
            self.data_detail.grid(column=0, row=6, rowspan=6, sticky='NSEW')
            ysb2 = ttk.Scrollbar(orient=tk.VERTICAL, command=self.data_detail.yview)
            xsb2 = ttk.Scrollbar(orient=tk.HORIZONTAL, command=self.data_detail.xview)
            self.data_detail.configure(yscroll=ysb2.set, xscroll=xsb2.set)

            # operation history of the selected dataset
            self.data_history = ttk.Treeview()
            self.data_history.heading('#0', text='Data history')
            self.data_history.grid(column=0, row=12, rowspan=2, sticky='NSEW')
            ysb3 = ttk.Scrollbar(orient=tk.VERTICAL, command=self.data_history.yview)
            xsb3 = ttk.Scrollbar(orient=tk.HORIZONTAL, command=self.data_history.xview)
            self.data_history.configure(yscroll=ysb3.set, xscroll=xsb3.set)

            self.notebook = ttk.Notebook(height=600, width=600)
            self.notebook.grid(column=1, row=0, rowspan=14, columnspan=4, sticky='NSEW')
            self.tabs = []

        if file_type == 'csv':  # sc-seq data
            scdata = scras.SCData.from_csv(os.path.expanduser(self.filename), data_name=self.fileNameEntryVar.get(),
                                        data_type='sc-seq', cell_axis=self.rowVar.get(),
                                        delimiter=self.delimiter.get(),
                                        rows_after_header_to_skip=self.rowHeader.get(),
                                        cols_after_header_to_skip=self.colHeader.get())

        elif file_type == 'mtx':  # sparse matrix
            pass  # to be implemented

        elif file_type == '10x':
            pass  # to be implemented

        if file_type != 'pickle':
            # filter the data
            if len(self.filterCellMinVar.get()) > 0 or len(self.filterCellMaxVar.get()) > 0 or \
                            len(self.filterGeneNonzeroVar.get()) > 0 or len(self.filterGeneMolsVar.get()) > 0:
                scdata.filter_scseq_data(
                    filter_cell_min=int(self.filterCellMinVar.get()) if len(self.filterCellMinVar.get()) > 0 else 0,
                    filter_cell_max=int(self.filterCellMaxVar.get()) if len(self.filterCellMaxVar.get()) > 0 else 0,
                    filter_gene_nonzero=int(self.filterGeneNonzeroVar.get()) if len(
                        self.filterGeneNonzeroVar.get()) > 0 else 0,
                    filter_gene_mols=int(self.filterGeneMolsVar.get()) if len(self.filterGeneMolsVar.get()) > 0 else 0)

            if self.normalizeVar.get() is True:
                scdata.normalize_scseq_data()

            if self.logTransform.get() is True:
                scdata.log_transform_scseq_data()

            if self.magicVar.get() is True:
                scdata.run_magic(self.mgCompVar.get(), self.mgRandomVar.get(), self.mgTVar.get(), self.mgKVar.get(),
                                 self.mgKaVar.get(), self.mgEpVar.get(), self.mgRescaleVar.get())

        else:  # pickled Wishbone object
            pass  # to be implemented

        self.data[self.fileNameEntryVar.get()] = scdata
        self.data_list.insert('', 'end', text=self.fileNameEntryVar.get() +
                                              ' (' + str(scdata.data.shape[0]) +
                                              ' x ' + str(scdata.data.shape[1]) + ')', open=True)

        # enable buttons
        self.fileMenu.entryconfig(5, state='normal')
        self.analysisMenu.entryconfig(0, state='normal')
        self.analysisMenu.entryconfig(1, state='normal')
        self.visMenu.entryconfig(0, state='normal')
        self.visMenu.entryconfig(1, state='normal')
        self.visMenu.entryconfig(2, state='normal')

        if len(self.data) > 1:
            self.fileMenu.entryconfig(4, state='normal')

        self.geometry('1000x650')

        self.import_options.destroy()

    def _deleteDataItem(self, event):
        self.data_detail.delete(*self.data_detail.get_children())
        self.data_history.delete(*self.data_history.get_children())

        for key in self.data_list.selection():
            self.curKey = key
            name = self.data_list.item(self.curKey, 'text').split(' (')[0]
            og_name = name if name.find(':') == -1 else name[:name.find(':')]
            # find the operation sequence of the parent dataset and use it to find the corresponding SCData object
            parentID = self.data_list.parent(self.curKey)
            if parentID:
                opseq = self._datafinder(self.data_list, parentID)
                og = self.data[og_name]
                scobj = scras.SCData.retrieve_data(og, opseq)
                scobj.datadict.pop(name)
            else:
                del self.data[og_name]

            self.data_list.delete(key)

    def _updateSelection(self, event):
        self.data_detail.delete(*self.data_detail.get_children())
        self.data_history.delete(*self.data_history.get_children())

        for key in self.data_list.selection():
            name = self.data_list.item(key)['text'].split(' (')[0]

            opseq = self._datafinder(self.data_list, key)
            og = self.data[opseq[0]]
            curdata = scras.SCData.retrieve_data(og, opseq)

            for op in curdata.operation.history:
                self.data_history.insert('', 'end', text=op, open=True)

            magic = True if 'MAGIC' in name else False

            if 'PCA' in name:
                for i in range(curdata.data.shape[1]):
                    if magic:
                        self.data_detail.insert('', 'end', text='MAGIC PC' + str(i + 1), open=True)
                    else:
                        self.data_detail.insert('', 'end', text='PC' + str(i + 1), open=True)

            elif 'DM' in name:
                for i in range(curdata.data.shape[1]):
                    if magic:
                        self.data_detail.insert('', 'end', text='MAGIC DC' + str(i + 1), open=True)
                    else:
                        self.data_detail.insert('', 'end', text='DC' + str(i + 1), open=True)

            else:
                for gene in curdata.data:
                    if magic:
                        self.data_detail.insert('', 'end', text='MAGIC ' + gene, open=True)
                    else:
                        self.data_detail.insert('', 'end', text=gene, open=True)

    def concatenate_data(self):
        self.concatOptions = tk.Toplevel()
        self.concatOptions.title("Concatenate data sets")

        tk.Label(self.concatOptions, text=u"New data set name:", fg="black", bg="white").grid(column=0, row=0)
        self.nameVar = tk.StringVar()
        tk.Entry(self.concatOptions, textvariable=self.nameVar).grid(column=1, row=0)

        self.colVar = tk.IntVar()
        tk.Radiobutton(self.concatOptions, text='Concatenate columns', variable=self.colVar, value=0).grid(column=0,
                                                                                                           row=1)
        tk.Radiobutton(self.concatOptions, text='Concatenate rows', variable=self.colVar, value=1).grid(column=1, row=1)

        self.joinVar = tk.BooleanVar()
        self.joinVar.set(True)
        tk.Checkbutton(self.concatOptions, text=u"Outer join", variable=self.joinVar).grid(column=0, row=2,
                                                                                           columnspan=2)

        tk.Button(self.concatOptions, text="Concatenate", command=self._concatenateData).grid(column=1, row=3)
        tk.Button(self.concatOptions, text="Cancel", command=self.concatOptions.destroy).grid(column=0, row=3)
        self.wait_window(self.concatOptions)

    def _concatenateData(self):
        to_concat = []
        selected = self.data_list.selection()
        path = self._datafinder(self.data_list, selected[0])
        og = self.data[path[0]]
        scobj = scras.SCData.retrieve_data(og, path)
        for key in selected[1:]:
            to_concat.append(self.data[self.data_list.item(key)['text'].split(' (')[0]])

        names = tuple(self.data_list.item(key)['text'].split(' (')[0] for key in selected)
        join = 'outer' if self.joinVar.get() is True else 'inner'
        scdata = scobj.concatenate_data(to_concat, names=names, axis=self.colVar.get(), join=join)

        self.data[self.nameVar.get()] = scdata
        self.data_list.insert('', 'end', text=self.nameVar.get() + ' (' + str(scdata.data.shape[0]) + ' x ' + str(
            scdata.data.shape[1]) + ')', open=True)

        for key in self.data_list.selection():
            self.data_list.delete(key)

        self.concatOptions.destroy()

    def save_data(self):
        for key in self.data_list.selection():
            path = self._datafinder(self.data_list, self.curKey)
            og = self.data[path[0]]
            scobj = scras.SCData.retrieve_data(og, path)
            name = self.data_list.item(key)['text'].split(' (')[0]
            pickleFileName = filedialog.asksaveasfilename(title=name + ': save data', defaultextension='.p',
                                                          initialfile=key)
            if pickleFileName is not None:
                scobj.save(pickleFileName)

    def save_plot(self):
        tab = self.notebook.index(self.notebook.select())
        default_name = self.notebook.tab(self.notebook.select(), "text")

        plotFileName = filedialog.asksaveasfilename(title='Save Plot', defaultextension='.png',
                                                         initialfile=default_name)
        if plotFileName is not None:
            self.tabs[tab][1].savefig(plotFileName)

    def run_dr(self):
        for key in self.data_list.selection():
            # pop up for parameters
            self.drOptions = tk.Toplevel()
            self.drOptions.resizable(False, False)
            self.drOptions.title(self.data_list.item(key)['text'].split(' (')[0] + ": Dimensionality reduction options")
            self.curKey = key

            # run PCA or not
            self.PCAVAR = tk.BooleanVar()
            self.PCAVAR.set(True)
            tk.Checkbutton(self.drOptions, text="PCA", command=self._updateDROptions,
                           variable=self.PCAVAR).grid(column=0, row=0, sticky='w')

            # how many PCA components
            pCompContainer = tk.Frame(self.drOptions)
            pCompContainer.grid(column=0, row=1, sticky='w')
            tk.Label(pCompContainer, text="Number of PCA components:", fg="black", bg="white").grid(column=0,
                                                                                                    row=0, sticky='w')
            self.pCompVar = tk.IntVar()
            self.pCompVar.set(100)
            self.pCompEntry = tk.Entry(pCompContainer, textvariable=self.pCompVar)
            self.pCompEntry.grid(column=1, row=0, sticky='w')

            # ranndomized PCA or no
            self.pRandomVar = tk.BooleanVar()
            self.pRandomVar.set(True)
            self.pRandom = tk.Checkbutton(self.drOptions, text="Randomized PCA (faster)", variable=self.pRandomVar)
            self.pRandom.grid(column=0, row=2, sticky='w')

            # separator
            ttk.Separator(self.drOptions, orient='horizontal').grid(column=0, row=3, sticky='ew', pady=8)

            # run DM or not
            self.DMVar = tk.BooleanVar()
            self.DMVar.set(False)
            tk.Checkbutton(self.drOptions, text="Diffusion components", command=self._updateDROptions,
                           variable=self.DMVar).grid(column=0, row=4, sticky='w')

            # how many diffsuion components
            dCompContainer = tk.Frame(self.drOptions)
            dCompContainer.grid(column=0, row=5, sticky='w')
            tk.Label(dCompContainer, text="Number of diffusion components:",
                     fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.dCompVar = tk.IntVar()
            self.dCompVar.set(10)
            self.dCompEntry = tk.Entry(dCompContainer, textvariable=self.dCompVar, state='disabled')
            self.dCompEntry.grid(column=1, row=0, sticky='w')

            # whats k
            dKContainer = tk.Frame(self.drOptions)
            dKContainer.grid(column=0, row=6, sticky='w')
            tk.Label(dKContainer, text="K:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.dKVar = tk.IntVar()
            self.dKVar.set(10)
            self.dKEntry = tk.Entry(dKContainer, textvariable=self.dKVar, state='disabled')
            self.dKEntry.grid(column=1, row=0, sticky='w')

            # whats ka
            dKaContainer = tk.Frame(self.drOptions)
            dKaContainer.grid(column=0, row=7, sticky='w')
            tk.Label(dKaContainer, text="Ka:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.dKaVar = tk.IntVar()
            self.dKaVar.set(0)
            self.dKaEntry = tk.Entry(dKaContainer, textvariable=self.dKaVar, state='disabled')
            self.dKaEntry.grid(column=1, row=0, sticky='w')

            # whats epsilon
            dEpContainer = tk.Frame(self.drOptions)
            dEpContainer.grid(column=0, row=8, sticky='w')
            tk.Label(dEpContainer, text="Epsilon:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.dEpVar = tk.IntVar()
            self.dEpVar.set(1)
            self.dEpEntry = tk.Entry(dEpContainer, textvariable=self.dEpVar, state='disabled')
            self.dEpEntry.grid(column=1, row=0, sticky='w')

            # which distance metric
            dDisContainer = tk.Frame(self.drOptions)
            dDisContainer.grid(column=0, row=9, sticky='w')
            tk.Label(dDisContainer, text="Distance metric:", fg="black", bg="white").grid(column=0, row=0)
            self.dDisVar = tk.StringVar()
            disChoices = {'euclidean', 'manhattan', 'correlation', 'cosine'}
            self.dDisVar.set('euclidean')
            self.dDisOption = tk.OptionMenu(dDisContainer, self.dDisVar, *disChoices)
            self.dDisOption.grid(column=1, row=0)
            self.dDisOption['state'] = 'disabled'

            dButtonContainer = tk.Frame(self.drOptions)
            dButtonContainer.grid(column=0, row=10, sticky='w')
            tk.Button(dButtonContainer, text="Cancel", command=self.drOptions.destroy).grid(column=0, row=0)
            tk.Button(dButtonContainer, text="Run", command=self._run_dr).grid(column=1, row=0)
            self.wait_window(self.drOptions)

    def _updateDROptions(self):
        if not self.PCAVAR.get():
            self.pCompEntry.config(state='disabled')
            self.pRandom.config(state='disabled')
        else:
            self.pCompEntry.config(state='normal')
            self.pRandom.config(state='normal')

        if self.DMVar.get():
            self.dCompEntry.config(state='normal')
            self.dKEntry.config(state='normal')
            self.dKaEntry.config(state='normal')
            self.dEpEntry.config(state='normal')
            self.dDisOption.config(state='normal')
            self.dDisOption['state'] = 'normal'
        else:
            self.dCompEntry.config(state='disabled')
            self.dKEntry.config(state='disabled')
            self.dKaEntry.config(state='disabled')
            self.dEpEntry.config(state='disabled')
            self.dDisOption.config(state='disabled')
            self.dDisOption['state'] = 'disabled'

    def _run_dr(self):
        path = self._datafinder(self.data_list, self.curKey)
        og = self.data[path[0]]
        scobj = scras.SCData.retrieve_data(og, path)

        if self.PCAVAR.get() and not self.DMVar.get():
            old_keys = set(scobj.datadict.keys())
            pcadata = scobj.run_pca(self.pCompVar.get(), self.pRandomVar.get())
            new_key = old_keys.symmetric_difference(set(scobj.datadict.keys()))
            new_key = list(new_key)[0]

            # insert the new key to the current tree view under the parent dataset
            self.data_list.insert(self.curKey, 'end', text=new_key + ' (' + str(pcadata.data.shape[0]) +
                                                           ' x ' + str(pcadata.data.shape[1]) + ')', open=True)

            # plot figure setup
            self.fig = plt.figure(figsize=[6, 6])
            gs = gridspec.GridSpec(1, 1)
            self.ax = self.fig.add_subplot(gs[0, 0])
            pcadata.plot_pca_variance_explained(self.pCompVar.get(), self.fig, self.ax, random=self.pRandomVar.get())

            self.ax.set_title(pcadata.name + 'Variance')
            self.ax.set_xlabel('PCA Components')
            self.ax.set_ylabel('Percent Total Variance')

            gs.tight_layout(self.fig)

            self.tabs.append([tk.Frame(self.notebook), self.fig])
            self.notebook.add(self.tabs[len(self.tabs) - 1][0], text="PCA")

            self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs) - 1][0])
            self.canvas.show()
            self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

            self.fileMenu.entryconfig(6, state='normal')

            self.currentPlot = 'pcavariance'

        elif not self.PCAVAR.get() and self.DMVar.get():
            old_keys = set(scobj.datadict.keys())
            dmdata = scobj.run_diffusion_map(self.dKVar.get(), self.dEpVar.get(),
                                             self.dDisVar.get(), self.dCompVar.get(), self.dKaVar.get())
            new_key = old_keys.symmetric_difference(set(scobj.datadict.keys()))
            new_key = list(new_key)[0]

            # insert the new key to the current tree view under the parent dataset
            self.data_list.insert(self.curKey, 'end', text=new_key + ' (' + str(dmdata.data.shape[0]) +
                                                           ' x ' + str(dmdata.data.shape[1]) + ')', open=True)
        elif self.PCAVAR.get() and self.DMVar.get():
            # run PCA first
            old_keys = set(scobj.datadict.keys())
            pcadata = scobj.run_pca(self.pCompVar.get(), self.pRandomVar.get())
            new_key = old_keys.symmetric_difference(set(scobj.datadict.keys()))
            new_key = list(new_key)[0]

            # insert the new key to the current tree view under the parent dataset
            self.curKey = self.data_list.insert(self.curKey, 'end', text=new_key + ' (' + str(pcadata.data.shape[0]) +
                                                                         ' x ' + str(pcadata.data.shape[1]) + ')',
                                                open=True)
            # plot PCA variance
            self.fig = plt.figure(figsize=[6, 6])
            gs = gridspec.GridSpec(1, 1)
            self.ax = self.fig.add_subplot(gs[0, 0])
            pcadata.plot_pca_variance_explained(self.pCompVar.get(), self.fig, self.ax, random=self.pRandomVar.get())

            self.ax.set_title(pcadata.name + 'Variance')
            self.ax.set_xlabel('PCA Components')
            self.ax.set_ylabel('Percent Total Variance')

            gs.tight_layout(self.fig)

            self.tabs.append([tk.Frame(self.notebook), self.fig])
            self.notebook.add(self.tabs[len(self.tabs) - 1][0], text="PCA")

            self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs) - 1][0])
            self.canvas.show()
            self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

            self.fileMenu.entryconfig(6, state='normal')

            self.currentPlot = 'pcavariance'

            # then run DM
            scobj = scobj.datadict[new_key]

            old_keys = set(scobj.datadict.keys())
            dmdata = scobj.run_diffusion_map(self.dKVar.get(), self.dEpVar.get(),
                                             self.dDisVar.get(), self.dCompVar.get(), self.dKaVar.get())
            new_key = old_keys.symmetric_difference(set(scobj.datadict.keys()))
            new_key = list(new_key)[0]

            # insert the new key to the current tree view under the parent dataset
            self.data_list.insert(self.curKey, 'end', text=new_key + ' (' + str(dmdata.data.shape[0]) +
                                                           ' x ' + str(dmdata.data.shape[1]) + ')', open=True)
        else:
            pass  # do nothing

        self.drOptions.destroy()

    def run_clustering(self):
        for key in self.data_list.selection():
            # pop up for parameters
            self.clusterOptions = tk.Toplevel()
            self.clusterOptions.title(self.data_list.item(key)['text'].split(' (')[0] + ": Clustering options")
            self.curKey = key

            tk.Label(self.clusterOptions, text="Phenograph options:", fg="black", bg="white").grid(column=0, row=0)

            # whats the number of nearest neighbors
            cKContainer = tk.Frame(self.clusterOptions)
            cKContainer.grid(column=0, row=1, sticky='w')
            tk.Label(cKContainer, text="Number of nearest neighbors:", fg="black", bg="white").grid(column=0,
                                                                                                    row=0, sticky='w')
            self.cKVar = tk.IntVar()
            self.cKVar.set(30)
            tk.Entry(cKContainer, textvariable=self.cKVar).grid(column=1, row=0, sticky='w')

            # whats the minimum cluster size
            cMinContainer = tk.Frame(self.clusterOptions)
            cMinContainer.grid(column=0, row=2, sticky='w')
            tk.Label(cMinContainer, text="Minimum cluster size:", fg="black", bg="white").grid(column=0,
                                                                                               row=0, sticky='w')
            self.cMinVar = tk.IntVar()
            self.cMinVar.set(10)
            tk.Entry(cMinContainer, textvariable=self.cMinVar).grid(column=1, row=0, sticky='w')

            # whats the distance metric
            cMinContainer = tk.Frame(self.clusterOptions)
            cMinContainer.grid(column=0, row=3, sticky='w')
            tk.Label(cMinContainer, text="Distance metric:", fg="black", bg="white").grid(column=0,
                                                                                                row=0, sticky='w')
            self.cChoiceVar = tk.StringVar()
            choices = {'euclidean', 'manhattan', 'correlation', 'cosine'}
            self.cChoiceVar.set('euclidean')
            tk.OptionMenu(cMinContainer, self.cChoiceVar, *choices).grid(column=1, row=0, sticky='w')

            # whats the number of jobs
            cNjobContainer = tk.Frame(self.clusterOptions)
            cNjobContainer.grid(column=0, row=4, sticky='w')
            tk.Label(cNjobContainer, text="Number of jobs:", fg="black", bg="white").grid(column=0,
                                                                                          row=0, sticky='w')
            self.cNjobVar = tk.IntVar()
            self.cNjobVar.set(-1)
            tk.Entry(cNjobContainer, textvariable=self.cNjobVar).grid(column=1, row=0, sticky='w')

            # whats the tolerance
            cToleContainer = tk.Frame(self.clusterOptions)
            cToleContainer.grid(column=0, row=5, sticky='w')
            tk.Label(cToleContainer, text="Tolerance:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.cToleVar = tk.DoubleVar()
            self.cToleVar.set(1e-3)
            tk.Entry(cToleContainer, textvariable=self.cToleVar).grid(column=1, row=0, sticky='w')

            # whats the Louvain time limit
            cLouvContainer = tk.Frame(self.clusterOptions)
            cLouvContainer.grid(column=0, row=6, sticky='w')
            tk.Label(cLouvContainer, text="Louvain time limit (s):", fg="black", bg="white").grid(column=0,
                                                                                                  row=0, sticky='w')
            self.cLouvVar = tk.IntVar()
            self.cLouvVar.set(2000)
            tk.Entry(cLouvContainer, textvariable=self.cLouvVar).grid(column=1, row=0, sticky='w')

            # whats the nn-method
            cNNContainer = tk.Frame(self.clusterOptions)
            cNNContainer.grid(column=0, row=7, sticky='w')
            tk.Label(cNNContainer, text="nn-method:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.cNNVar = tk.StringVar()
            nn_choices = {'brute force', 'kdtree'}
            self.cNNVar.set('kdtree')
            tk.OptionMenu(cNNContainer, self.cNNVar, *nn_choices).grid(column=1, row=0, sticky='w')

            # some yes or no questions
            cCheckContainer = tk.Frame(self.clusterOptions)
            cCheckContainer.grid(column=0, row=8)
            self.cPruneVar = tk.BooleanVar()
            self.cPruneVar.set(False)
            tk.Checkbutton(cCheckContainer, text="Prune", variable=self.cPruneVar).grid(column=0, row=0)
            self.cDirectedVar = tk.BooleanVar()
            self.cDirectedVar.set(False)
            tk.Checkbutton(cCheckContainer, text="Directed Graph", variable=self.cDirectedVar).grid(column=1, row=0)
            self.cJaccVar = tk.BooleanVar()
            self.cJaccVar.set(True)
            tk.Checkbutton(cCheckContainer, text="Jaccard Metric", variable=self.cJaccVar).grid(column=2, row=0)

            cButtonContainer = tk.Frame(self.clusterOptions)
            cButtonContainer.grid(column=0, row=9)
            tk.Button(cButtonContainer, text="Cancel", command=self.clusterOptions.destroy).grid(column=0, row=0)
            tk.Button(cButtonContainer, text="Run", command=self._run_clustering).grid(column=1, row=0)
            self.wait_window(self.clusterOptions)

    def _run_clustering(self):
        self.clusterOptions.destroy()

        path = self._datafinder(self.data_list, self.curKey)
        og = self.data[path[0]]
        scobj = scras.SCData.retrieve_data(og, path)

        self.phenoProgress = tk.Toplevel()
        self.msgVar = tk.StringVar()
        self.msgVar.set("starting PhenoGraph...")
        self.phenoProgress.title(scobj.name + ': running PhenoGraph')
        self.msgLb = tk.Label(self.phenoProgress, textvariable=self.msgVar).pack()

        self.phenoProgress.update()

        # run tsne with default setting
        tsnedata = scobj.run_tsne()

        self.msgVar.set("running PhenoGraph...")
        self.phenoProgress.update()
        communities, Q = scobj.run_phenograph(k=self.cKVar.get(), directed=self.cDirectedVar.get(),
                                        prune=self.cPruneVar.get(), min_cluster_size=self.cMinVar.get(),
                                        jaccard=self.cJaccVar.get(), dis_metric=self.cChoiceVar.get(),
                                        n_jobs=self.cNjobVar.get(), q_tol=self.cToleVar.get(),
                                        louvain_time_limit=self.cLouvVar.get(), nn_method=self.cNNVar.get())
        if min(set(communities)) == 0:
            communities = [x+1 for x in communities]
        color = pd.Series(communities)

        # plot figure setup
        toPlot = tsnedata.assign(com=pd.Series(color).values)
        clusterRec = {}
        self.fig = plt.figure(figsize=[6, 6])
        gs = gridspec.GridSpec(1, 1)
        self.ax = self.fig.add_subplot(gs[0, 0])

        # plot tsne using communities to label color
        self.msgVar.set("plotting data points...")
        self.phenoProgress.update()

        scras.SCData.plot_tsne(tsnedata, self.fig, self.ax, color=color)
        self.ax.set_title(scobj.name)
        self.ax.set_xlabel('tSNE1')
        self.ax.set_ylabel('tSNE2')

        """
        # position cluster number at cluster center
        for index, row in toPlot.iterrows():
            if row['com'] in clusterRec:
                count = clusterRec[row['com']][2]
                new1 = (clusterRec[row['com']][0] * count + row['tSNE1']) / (count + 1)
                new2 = (clusterRec[row['com']][1] * count + row['tSNE2']) / (count + 1)
                clusterRec[row['com']] = [new1, new2, count + 1]
            else:
                clusterRec[row['com']] = [row['tSNE1'], row['tSNE2'], 1]

        for key in clusterRec:
            x, y = clusterRec[key][0], clusterRec[key][1]
            self.ax.annotate(str(int(key)), (x, y), fontsize=20, weight='bold', color='#777777')
        """

        # add figure to the GUI
        gs.tight_layout(self.fig)

        self.tabs.append([tk.Frame(self.notebook), self.fig])
        self.notebook.add(self.tabs[len(self.tabs) - 1][0], text="PhenoGraph")

        self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs) - 1][0])
        self.canvas.show()
        self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

        # enable plot saving
        self.fileMenu.entryconfig(6, state='normal')

        self.currentPlot = 'phenograph'
        self.phenoProgress.destroy()

        numCluster = np.max(communities)
        communities = [int(i) for i in communities]
        diff = 1 - min(communities)
        communities = [str(i + diff) for i in communities]

        self.phenoResult = tk.Toplevel()
        self.phenoResult.title(scobj.name + " PhenoGraph Results")
        tk.Label(self.phenoResult, text="Number of clusters: " + str(numCluster),
                 fg="black", bg="white").grid(column=0, row=1, sticky='w')
        tk.Label(self.phenoResult, text="Modularity score: " + str(Q), fg="black", bg="white").grid(column=0,
                                                                                                    row=2, sticky='w')
        tk.Button(self.phenoResult, text="Ok", command=self.phenoResult.destroy).grid(column=0, row=3)
        tk.Button(self.phenoResult, text="Save communities as CSV",
                  command=lambda: self.saveCSV(scobj, pd.Series(communities))).grid(column=1, row=3)
        self.phenoResult.update()
        self.wait_window(self.phenoResult)

    def run_gea(self):
        # pop up a new window reporting the result and provide visualization
        pass  # to be implemented

    def tsne(self):
        for key in self.data_list.selection():
            # pop up for parameters
            self.tsneOptions = tk.Toplevel()
            self.tsneOptions.resizable(False, False)
            self.tsneOptions.title(self.data_list.item(key)['text'].split(' (')[0] + ": tSNE plotting options")
            self.curKey = key

            # what is the perplexity
            tPerpContainer = tk.Frame(self.tsneOptions)
            tPerpContainer.grid(column=0, row=0, sticky='w')
            tk.Label(tPerpContainer, text="Perplexity:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.tPerpVar = tk.IntVar()
            self.tPerpVar.set(30)
            self.tPerpEntry = tk.Entry(tPerpContainer, textvariable=self.tPerpVar)
            self.tPerpEntry.grid(column=1, row=0, sticky='w')

            # number of iterations
            tIterContainer = tk.Frame(self.tsneOptions)
            tIterContainer.grid(column=0, row=1, sticky='w')
            tk.Label(tIterContainer, text="Number of iterations:", fg="black",
                     bg="white").grid(column=0, row=0, sticky='w')
            self.tIterVar = tk.IntVar()
            self.tIterVar.set(1000)
            self.tIterEntry = tk.Entry(tIterContainer, textvariable=self.tIterVar)
            self.tIterEntry.grid(column=1, row=0, sticky='w')

            # value of theta
            tThetaContainer = tk.Frame(self.tsneOptions)
            tThetaContainer.grid(column=0, row=2, sticky='w')
            tk.Label(tThetaContainer, text="Theta:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.tThetaVar = tk.DoubleVar()
            self.tThetaVar.set(0.5)
            self.tThetaEntry = tk.Entry(tThetaContainer, textvariable=self.tThetaVar)
            self.tThetaEntry.grid(column=1, row=0, sticky='w')

            # color of plot
            tColorContainer = tk.Frame(self.tsneOptions)
            tColorContainer.grid(column=0, row=3, sticky='w')
            tk.Label(tColorContainer, text="Color:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.tColorVar = tk.StringVar()
            self.tColorVar.set('blue')
            self.tColorEntry = tk.Entry(tColorContainer, textvariable=self.tColorVar)
            self.tColorEntry.grid(column=1, row=0, sticky='w')

            tButtonContainer = tk.Frame(self.tsneOptions)
            tButtonContainer.grid(column=0, row=10, sticky='w')
            tk.Button(tButtonContainer, text="Cancel", command=self.tsneOptions.destroy).grid(column=0, row=0)
            tk.Button(tButtonContainer, text="Run", command=self._tsne).grid(column=1, row=0)
            self.wait_window(self.tsneOptions)

    def _tsne(self):
        path = self._datafinder(self.data_list, self.curKey)
        og = self.data[path[0]]
        scobj = scras.SCData.retrieve_data(og, path)

        tsnedata = scobj.run_tsne(self.tPerpVar.get(), self.tIterVar.get(), self.tThetaVar.get())

        # plot figure setup
        self.fig = plt.figure(figsize=[6, 6])
        gs = gridspec.GridSpec(1, 1)
        self.ax = self.fig.add_subplot(gs[0, 0])
        color = self.tColorVar.get()

        scras.SCData.plot_tsne(tsnedata, self.fig, self.ax, color=color)
        self.ax.set_title(scobj.name + ' (color =' + color + ')')
        self.ax.set_xlabel('tSNE1')
        self.ax.set_ylabel('tSNE2')

        gs.tight_layout(self.fig)

        self.tabs.append([tk.Frame(self.notebook), self.fig])
        self.notebook.add(self.tabs[len(self.tabs) - 1][0], text="tSNE")

        self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs) - 1][0])
        self.canvas.show()
        self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

        self.fileMenu.entryconfig(6, state='normal')

        self.currentPlot = 'tsne'

        self.tsneOptions.destroy()

    def scatter_plot(self):
        for key in self.data_list.selection():
            # pop up for parameters
            self.scatterOptions = tk.Toplevel()
            self.scatterOptions.resizable(False, False)
            self.scatterOptions.title(self.data_list.item(key)['text'].split(' (')[0] + ": scatter plotting options")
            self.curKey = key

            # what is x
            sXContainer = tk.Frame(self.scatterOptions)
            sXContainer.grid(column=0, row=0, sticky='w')
            tk.Label(sXContainer, text="X:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.sXVar = tk.StringVar()
            self.sXVar.set('')
            self.sXEntry = tk.Entry(sXContainer, textvariable=self.sXVar)
            self.sXEntry.grid(column=1, row=0, sticky='w')

            # what is y
            sYContainer = tk.Frame(self.scatterOptions)
            sYContainer.grid(column=0, row=1, sticky='w')
            tk.Label(sYContainer, text="Y:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.sYVar = tk.StringVar()
            self.sYVar.set('')
            self.sYEntry = tk.Entry(sYContainer, textvariable=self.sYVar)
            self.sYEntry.grid(column=1, row=0, sticky='w')

            # what is z
            sZContainer = tk.Frame(self.scatterOptions)
            sZContainer.grid(column=0, row=2, sticky='w')
            tk.Label(sZContainer, text="Z:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.sZVar = tk.StringVar()
            self.sZVar.set('')
            self.sZEntry = tk.Entry(sZContainer, textvariable=self.sZVar)
            self.sZEntry.grid(column=1, row=0, sticky='w')

            # color of plot
            sColorContainer = tk.Frame(self.scatterOptions)
            sColorContainer.grid(column=0, row=3, sticky='w')
            tk.Label(sColorContainer, text="Color:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.sColorVar = tk.StringVar()
            self.sColorVar.set('blue')
            self.sColorEntry = tk.Entry(sColorContainer, textvariable=self.sColorVar)
            self.sColorEntry.grid(column=1, row=0, sticky='w')

            sButtonContainer = tk.Frame(self.scatterOptions)
            sButtonContainer.grid(column=0, row=10)
            tk.Button(sButtonContainer, text="Cancel", command=self.scatterOptions.destroy).grid(column=0, row=0)
            tk.Button(sButtonContainer, text="Run", command=self._scatter_plot).grid(column=1, row=0)
            self.wait_window(self.scatterOptions)

    def _scatter_plot(self):
        path = self._datafinder(self.data_list, self.curKey)
        og = self.data[path[0]]
        scobj = scras.SCData.retrieve_data(og, path)

        # plot figure setup
        self.fig = plt.figure(figsize=[6, 6])
        gs = gridspec.GridSpec(1, 1)
        if self.sZVar.get() != '':  # 3D graph
            self.ax = self.fig.add_subplot(gs[0, 0], projection='3d')
            toplot = (self.sXVar.get(), self.sYVar.get(), self.sZVar.get())
        else:
            self.ax = self.fig.add_subplot(gs[0, 0])
            toplot = (self.sXVar.get(), self.sYVar.get())
        color = self.sColorVar.get()

        scobj.scatter_gene_expression(toplot, color=self.sColorVar.get(), fig=self.fig, ax=self.ax)
        self.ax.set_title(scobj.name + ' (color =' + color + ')')
        self.ax.set_xlabel(toplot[0])
        self.ax.set_ylabel(toplot[1])
        if len(toplot) == 3:
            self.ax.set_zlabel(toplot[2])

        gs.tight_layout(self.fig)

        self.tabs.append([tk.Frame(self.notebook), self.fig])
        self.notebook.add(self.tabs[len(self.tabs) - 1][0], text="Scatter")

        self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs) - 1][0])
        self.canvas.show()
        self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

        self.fileMenu.entryconfig(5, state='normal')

        self.currentPlot = 'scatter'

        if self.sZVar.get() != '':
            self.ax.mouse_init()

        self.scatterOptions.destroy()

    def gene_expression(self):
        for key in self.data_list.selection():
            # pop up for parameters
            self.geOptions = tk.Toplevel()
            self.geOptions.resizable(False, False)
            self.geOptions.title(self.data_list.item(key)['text'].split(' (')[0] + ": gene expression options")
            self.curKey = key

            # which feature
            geXContainer = tk.Frame(self.geOptions)
            geXContainer.grid(column=0, row=0, sticky='w')
            tk.Label(geXContainer, text="feature/gene:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            self.geXVar = tk.StringVar()
            self.geXVar.set('')
            self.geXEntry = tk.Entry(geXContainer, textvariable=self.geXVar)
            self.geXEntry.grid(column=1, row=0, sticky='w')

            geButtonContainer = tk.Frame(self.geOptions)
            geButtonContainer.grid(column=0, row=10)
            tk.Button(geButtonContainer, text="Cancel", command=self.geOptions.destroy).grid(column=0, row=0)
            tk.Button(geButtonContainer, text="Run", command=self._gene_expression).grid(column=1, row=0)

            self.wait_window(self.geOptions)

    def _gene_expression(self):
        path = self._datafinder(self.data_list, self.curKey)
        og = self.data[path[0]]
        scobj = scras.SCData.retrieve_data(og, path)

        tsnedata = scobj.run_tsne()

        # plot figure setup
        self.fig = plt.figure(figsize=[6, 6])
        gs = gridspec.GridSpec(1, 1)
        self.ax = self.fig.add_subplot(gs[0, 0])

        # store the expression of selected gene as a pd.Series
        feature = self.geXVar.get()
        if feature not in scobj.data.columns.values:
            raise RuntimeError("feature not in values")
        feature = scobj.data[feature]
        print(feature)

        scras.SCData.plot_tsne(tsnedata, self.fig, self.ax, color=feature, ge=True)
        self.ax.set_title(scobj.name + ' (feature =' + self.geXVar.get() + ')')
        self.ax.set_xlabel('tSNE1')
        self.ax.set_ylabel('tSNE2')

        gs.tight_layout(self.fig)

        self.tabs.append([tk.Frame(self.notebook), self.fig])
        self.notebook.add(self.tabs[len(self.tabs) - 1][0], text="tSNE")

        self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs) - 1][0])
        self.canvas.show()
        self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

        self.fileMenu.entryconfig(6, state='normal')

        self.currentPlot = 'tsne'

        self.geOptions.destroy()

    def saveCSV(self, scdata, col):
        self.phenoResult.destroy()

        toSave = scdata.data.assign(com=pd.Series(col).values)
        csvFile = filedialog.asksaveasfile(title='Save as CSV', defaultextension='.csv', mode='w')

        if csvFile:
            clusters = []
            cell_map = {}
            for index, row in toSave.iterrows():
                if row['com'] in clusters:
                    cell_map[row['com']].append(index)
                else:
                    clusters.append(row['com'])
                    cell_map[row['com']] = [index]

            clusters = sorted(clusters)
        else:
            return

        writer = csv.writer(csvFile, delimiter=',')
        writer.writerow(clusters)
        for clus in clusters:
            writer.writerow(cell_map[clus])

        csvFile.close()

    @staticmethod
    def _keygen(name: str, op: str, params: list):
        par = params[0] if len(params) == 1 else "-".join(params)
        key = name + ':' + op + ':' + par

        return key

    @staticmethod
    def _datafinder(tktree, curselection):
        curID = curselection
        selname = tktree.item(curID)['text']
        selname = selname[:selname.find('(') - 1]
        path = [selname]
        parentID = tktree.parent(curID)

        while parentID:
            parentname = tktree.item(parentID)['text']
            parentname = parentname[:parentname.find('(') - 1]
            path.insert(0, parentname)
            curID = parentID
            parentID = tktree.parent(curID)

        return path

    def quit_scras(self):
        pass  # to be implemented


def launch():
    app = SCRASGui(None)
    if platform.system() == 'Darwin':
        app.focus_force()
    elif platform.system() == 'Windows':
        app.lift()
        app.call('wm', 'attributes', '.', '-topmost', True)
        app.after_idle(app.call, 'wm', 'attributes', '.', '-topmost', False)
    elif platform.system() == 'Linux':
        app.focus_force()

    app.title('SCRAS')

    while True:
        try:
            app.mainloop()
            break
        except UnicodeDecodeError:
            pass


if __name__ == "__main__":
    launch()
