#!/usr/local/bin/python3

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import reduce, partial
import os
import sys
import platform
import pandas as pd
import tkinter as tk
import numpy as np
from tkinter import filedialog, ttk
import phenograph
import csv

sys.path.insert(0, '/Users/vincentliu/PycharmProjects/magic/src/magic')
import mg_new as mg


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
        filename = filedialog.askopenfilename(title='Load data file', initialdir='~/.magic/data')

        if filename:
            import_options = tk.Toplevel()
            import_options.resizable(False, False)
            import_options.title('Import options')

            fileNameContainer = tk.Frame(import_options)
            fileNameContainer.grid(column=0, row=0, sticky='w')
            tk.Label(fileNameContainer, text="File name: ", pady=5).grid(column=0, row=0, sticky='w')
            tk.Label(fileNameContainer, text=filename.split('/')[-1]).grid(column=1, row=0, sticky='w')

            nameEntryContainer = tk.Frame(import_options)
            nameEntryContainer.grid(column=0, row=1, sticky='w')
            tk.Label(nameEntryContainer, text="Data name: ").grid(column=0, row=0, sticky='w')
            fileNameEntryVar = tk.StringVar()
            fileNameEntryVar.set('Data ' + str(len(self.data) + 1))
            tk.Entry(nameEntryContainer, textvariable=fileNameEntryVar).grid(column=1, row=0, sticky='w')

            delimiterContainer = tk.Frame(import_options)
            delimiterContainer.grid(column=0, row=2, sticky='w')
            tk.Label(delimiterContainer, text="Delimiter: ").grid(column=0, row=0, sticky='w')
            delimiter = tk.StringVar()
            delimiter.set(',')
            tk.Entry(delimiterContainer, textvariable=delimiter).grid(column=1, row=0, sticky='w')

            rowSelectionContainer = tk.Frame(import_options)
            rowSelectionContainer.grid(column=0, row=3, sticky='w')
            tk.Label(rowSelectionContainer, text="Rows:", fg="black", bg="white").grid(column=0, row=0, sticky='w')
            rowVar = tk.IntVar()
            rowVar.set(0)
            tk.Radiobutton(rowSelectionContainer, text="Cells", variable=rowVar, value=0).grid(column=1, row=0, sticky='W')
            tk.Radiobutton(rowSelectionContainer, text="Genes", variable=rowVar, value=1).grid(column=2, row=0, sticky='W')

            skipSelectionContainer = tk.Frame(import_options)
            skipSelectionContainer.grid(column=0, row=4, sticky='w')
            tk.Label(skipSelectionContainer, text="Number of additional rows/columns to skip after gene/cell names").grid(
                column=0, row=0, columnspan=3, sticky='W')

            numRowContainer = tk.Frame(skipSelectionContainer)
            numRowContainer.grid(column=0, row=1, sticky='W')
            tk.Label(numRowContainer, text="Number of rows:").grid(column=0, row=0, sticky='W')
            rowHeader = tk.IntVar()
            rowHeader.set(0)
            tk.Entry(numRowContainer, textvariable=rowHeader).grid(column=1, row=0, sticky='W')

            numColContainer = tk.Frame(skipSelectionContainer)
            numColContainer.grid(column=0, row=2, sticky='W')
            tk.Label(numColContainer, text="Number of columns:").grid(column=0, row=0, sticky='W')
            colHeader = tk.IntVar()
            colHeader.set(0)
            tk.Entry(numColContainer, textvariable=colHeader).grid(column=1, row=0, sticky='W')

            tk.Button(import_options, text="Compute data statistics",
                      command=partial(self.showRawDataDistributions, file_type='csv')).grid(column=0, row=5,
                                                                                            sticky='W', padx=8)

            ttk.Separator(import_options, orient='horizontal').grid(column=0, row=6, sticky='ew', pady=8)

            tk.Label(import_options, text="Filtering options (leave blank if no filtering)",
                     fg="black", bg="white", font="bold").grid(column=0, row=7, sticky='w')

            # filter parameters
            molPerCellContainer = tk.Frame(import_options)
            molPerCellContainer.grid(column=0, row=8, sticky='w')
            filterCellMinVar = tk.StringVar()
            tk.Label(molPerCellContainer, text="Filter by molecules per cell  Min:",
                     fg="black", bg="white").grid(column=0, row=0, sticky='w')
            tk.Entry(molPerCellContainer, textvariable=filterCellMinVar).grid(column=1, row=0, sticky='w')
            filterCellMaxVar = tk.StringVar()
            tk.Label(molPerCellContainer, text="Max:",
                     fg="black", bg="white").grid(column=0, row=1, sticky='E')
            tk.Entry(molPerCellContainer, textvariable=filterCellMaxVar).grid(column=1, row=1, sticky='w')

            cellPerGeneContainer = tk.Frame(import_options)
            cellPerGeneContainer.grid(column=0, row=9, sticky='w')
            filterGeneNonzeroVar = tk.StringVar()
            tk.Label(cellPerGeneContainer, text="Filter by nonzero cells per gene  Min:", fg="black", bg="white").grid(
                column=0, row=0, sticky='w')
            tk.Entry(cellPerGeneContainer, textvariable=filterGeneNonzeroVar).grid(column=1, row=0, sticky='w')

            molPerGeneContainer = tk.Frame(import_options)
            molPerGeneContainer.grid(column=0, row=10, sticky='w')
            filterGeneMolsVar = tk.StringVar()
            tk.Label(molPerGeneContainer, text="Filter by molecules per gene. Min:",
                     fg="black", bg="white").grid(column=0, row=0, sticky='w')
            tk.Entry(molPerGeneContainer, textvariable=filterGeneMolsVar).grid(column=1, row=0, sticky='w')

            # horizontal separator
            ttk.Separator(import_options, orient='horizontal').grid(column=0, row=11, sticky='ew', pady=8)

            tk.Label(import_options, text="Data pre-processing options",
                     fg="black", bg="white").grid(column=0, row=12, sticky='w')

            checkButtonContainer = tk.Frame(import_options)
            checkButtonContainer.grid(column=0, row=13, sticky='w')
            # normalize
            normalizeVar = tk.BooleanVar()
            normalizeVar.set(True)
            tk.Checkbutton(checkButtonContainer, text="Normalize by library size",
                           variable=normalizeVar).grid(column=0, row=0, sticky='w')
            # log transform
            logTransform = tk.BooleanVar()
            logTransform.set(True)
            tk.Checkbutton(checkButtonContainer,
                           text="Log-transform data", variable=logTransform).grid(column=1, row=0, sticky='w')
            # MAGIC
            magicVar = tk.BooleanVar()
            magicVar.set(True)
            tk.Checkbutton(checkButtonContainer,
                           text="Run MAGIC on the data", variable=magicVar).grid(column=3, row=0, sticky='w')

            # MAGIC options
            tk.Label(import_options, text="MAGIC options").grid(column=0, row=14, pady=8, sticky='w')

            mgPCACompContainer = tk.Frame(import_options)
            mgPCACompContainer.grid(column=0, row=15, sticky='w')
            tk.Label(mgPCACompContainer, text="Number of PCA components:", fg="black", bg="white").grid(column=0, row=0)
            self.mgCompVar = tk.IntVar()
            self.mgCompVar.set(20)
            tk.Entry(mgPCACompContainer, textvariable=self.mgCompVar, state='disabled').grid(column=1, row=0)
            self.randomVar = tk.BooleanVar()
            self.randomVar.set(True)
            tk.Checkbutton(mgPCACompContainer,
                           text="Randomized PCA", variable=self.randomVar,
                           state='disabled').grid(column=1, row=1, sticky='W')

            mgTContainer = tk.Frame(import_options)
            mgTContainer.grid(column=0, row=16, sticky='W')
            tk.Label(mgTContainer, text="t:", fg="black", bg="white").grid(column=0, row=0, sticky='W')
            self.tVar = tk.IntVar()
            self.tVar.set(6)
            tk.Entry(mgTContainer, textvariable=self.tVar, state='disabled').grid(column=1, row=0, sticky='W')

            mgKContainer = tk.Frame(import_options)
            mgKContainer.grid(column=0, row=17, sticky='w')
            tk.Label(mgKContainer, text="k:", fg="black", bg="white").grid(column=0, row=0, sticky='W')
            self.kVar = tk.IntVar()
            self.kVar.set(30)
            tk.Entry(mgKContainer, textvariable=self.kVar, state='disabled').grid(column=1, row=0, sticky='W')

            mgKaContainer = tk.Frame(import_options)
            mgKaContainer.grid(column=0, row=18, sticky='w')
            tk.Label(mgKaContainer, text="ka:", fg="black", bg="white").grid(column=0, row=0, sticky='W')
            self.autotuneVar = tk.IntVar()
            self.autotuneVar.set(10)
            tk.Entry(mgKaContainer, textvariable=self.autotuneVar, state='disabled').grid(column=1, row=0, sticky='W')

            mgEpContainer = tk.Frame(import_options)
            mgEpContainer.grid(column=0, row=19, sticky='w')
            tk.Label(mgEpContainer, text="Epsilon:", fg="black", bg="white").grid(column=0, row=0, sticky='W')
            self.epsilonVar = tk.IntVar()
            self.epsilonVar.set(1)
            tk.Entry(mgEpContainer, textvariable=self.epsilonVar, state='disabled').grid(column=1, row=0, sticky='W')
            tk.Label(mgEpContainer, text="(0 is the uniform kernel)",
                     fg="black", bg="white").grid(column=2, row=0)

            mgRescaleContainer = tk.Frame(import_options)
            mgRescaleContainer.grid(column=0, row=20, sticky='w')
            self.rescaleVar = tk.IntVar()
            self.rescaleVar.set(99)
            tk.Label(mgRescaleContainer, text="Rescale data to ",
                     fg="black", bg="white").grid(column=0, row=0, sticky='w')
            tk.Entry(mgRescaleContainer, textvariable=self.rescaleVar,
                     state='disabled').grid(column=1, row=0, sticky='w')
            tk.Label(mgRescaleContainer, text=" percentile (use 0 for log-transformed data)",
                     fg="black", bg="white").grid(column=2, row=0, sticky='w')

            finalButtonContainer = tk.Frame(import_options)
            finalButtonContainer.grid(column=0, row=21)
            tk.Button(finalButtonContainer, text="Cancel", command=import_options.destroy).grid(column=0, row=0, padx=20)
            tk.Button(finalButtonContainer, text="Load",
                      command=partial(self.process_data, file_type='csv')).grid(column=1, row=0, padx=20)

            self.wait_window(import_options)

    def load_mtx(self):
        pass

    def load_10x(self):
        pass

    def load_pickle(self):
        pass

    def showRawDataDistributions(self):
        pass

    def process_data(self, file_type):
        pass

    def concatenate_data(self):
        pass

    def save_data(self):
        pass

    def save_plot(self):
        pass

    def run_dr(self):
        pass

    def run_clustering(self):
        pass

    def run_gea(self):
        pass

    def tsne(self):
        pass

    def scatter_plot(self):
        pass

    def quit_scras(self):
        pass


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
