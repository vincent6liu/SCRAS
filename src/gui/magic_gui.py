#!/usr/local/bin/python3

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from functools import reduce, partial
import magic
import os
import sys
import platform
import pandas as pd
import tkinter as tk
import numpy as np
from tkinter import filedialog, ttk
import pickle
import random
import phenograph
import time
import csv

class magic_gui(tk.Tk):
    def __init__(self,parent):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.grid()
        self.vals = None
        self.currentPlot = None
        self.data = {}

        #set up menu bar
        self.menubar = tk.Menu(self)
        self.fileMenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.fileMenu)
        self.fileMenu.add_command(label="Load csv file", command=self.loadCSV)
        self.fileMenu.add_command(label="Load sparse data file", command=self.loadMTX)
        self.fileMenu.add_command(label="Load 10x file", command=self.load10x)
        self.fileMenu.add_command(label="Load saved session from pickle file", command=self.loadPickle)
        self.fileMenu.add_command(label="Save data", state='disabled', command=self.saveData)
        self.fileMenu.add_command(label="Save plot", state='disabled', command=self.savePlot)
        self.fileMenu.add_command(label="Concatenate datasets", state='disabled', command=self.concatenateData)
        self.fileMenu.add_command(label="Exit", command=self.quitMAGIC)

        self.analysisMenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Analysis", menu=self.analysisMenu)
        self.analysisMenu.add_command(label="Principal Component Analysis", state='disabled', command=self.runPCA)
        self.analysisMenu.add_command(label="MAGIC", state='disabled', command=self.runMagic)
        self.analysisMenu.add_command(label="PhenoGraph", state='disabled', command=self.runPhenoGraph)

        self.visMenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Visualization", menu=self.visMenu)
        self.visMenu.add_command(label="tSNE", state='disabled', command=self.runTSNE)
        self.visMenu.add_command(label="Diffusion map", state='disabled', command=self.runDM)
        self.visMenu.add_command(label="Scatter plot", state='disabled', command=self.scatterPlot)
        # self.visMenu.add_command(label="PCA-variance plot", state='disabled', command=self.plotPCAVariance)
        
        self.config(menu=self.menubar)

        #intro screen
        tk.Label(self, text=u"MAGIC", font=('Helvetica', 48), fg="black", bg="white", padx=100, pady=20).grid(row=0)
        tk.Label(self, text=u"Markov Affinity-based Graph Imputation of Cells", font=('Helvetica', 25), fg="black", bg="white", padx=100, pady=40).grid(row=1)
        tk.Label(self, text=u"To get started, select a data file by clicking File > Load Data", fg="black", bg="white", padx=100, pady=25).grid(row=2)

        #update
        self.protocol('WM_DELETE_WINDOW', self.quitMAGIC)
        self.grid_columnconfigure(0,weight=1)
        self.resizable(True,True)
        self.update()
        self.geometry(self.geometry())       
        self.focus_force()

    def loadCSV(self):
        self.dataFileName = filedialog.askopenfilename(title='Load data file', initialdir='~/.magic/data')
        if(self.dataFileName != ""):
            #pop up data options menu
            self.fileInfo = tk.Toplevel()
            self.fileInfo.title("Data options")
            tk.Label(self.fileInfo, text=u"File name: ").grid(column=0, row=0)
            tk.Label(self.fileInfo, text=self.dataFileName.split('/')[-1]).grid(column=1, row=0)

            tk.Label(self.fileInfo,text=u"Name:" ,fg="black",bg="white").grid(column=0, row=1)
            self.fileNameEntryVar = tk.StringVar()
            self.fileNameEntryVar.set('Data ' + str(len(self.data)))
            tk.Entry(self.fileInfo, textvariable=self.fileNameEntryVar).grid(column=1,row=1)

            tk.Label(self.fileInfo, text=u"Delimiter:").grid(column=0, row=2)
            self.delimiter = tk.StringVar()
            self.delimiter.set(',')
            tk.Entry(self.fileInfo, textvariable=self.delimiter).grid(column=1, row=2)

            tk.Label(self.fileInfo, text=u"Rows:", fg="black",bg="white").grid(column=0, row=3)
            self.rowVar = tk.IntVar()
            self.rowVar.set(0)
            tk.Radiobutton(self.fileInfo, text="Cells", variable=self.rowVar, value=0).grid(column=1, row=3)
            tk.Radiobutton(self.fileInfo, text="Genes", variable=self.rowVar, value=1).grid(column=2, row=3)

            tk.Label(self.fileInfo, text=u"Number of additional rows/columns to skip after gene/cell names").grid(column=0, row=4, columnspan=3)
            tk.Label(self.fileInfo, text=u"Number of rows:").grid(column=0, row=5)
            self.rowHeader = tk.IntVar()
            self.rowHeader.set(0)
            tk.Entry(self.fileInfo, textvariable=self.rowHeader).grid(column=1, row=5)

            tk.Label(self.fileInfo, text=u"Number of columns:").grid(column=0, row=6)
            self.colHeader = tk.IntVar()
            self.colHeader.set(0)
            tk.Entry(self.fileInfo, textvariable=self.colHeader).grid(column=1, row=6)


            tk.Button(self.fileInfo, text="Compute data statistics", command=partial(self.showRawDataDistributions, file_type='csv')).grid(column=1, row=7)

            #filter parameters
            self.filterCellMinVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by molecules per cell. Min:" ,fg="black",bg="white").grid(column=0, row=8)
            tk.Entry(self.fileInfo, textvariable=self.filterCellMinVar).grid(column=1,row=8)
            
            self.filterCellMaxVar = tk.StringVar()
            tk.Label(self.fileInfo, text=u" Max:" ,fg="black",bg="white").grid(column=2, row=8)
            tk.Entry(self.fileInfo, textvariable=self.filterCellMaxVar).grid(column=3,row=8)
            
            self.filterGeneNonzeroVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by nonzero cells per gene. Min:" ,fg="black",bg="white").grid(column=0, row=9)
            tk.Entry(self.fileInfo, textvariable=self.filterGeneNonzeroVar).grid(column=1,row=9)
            
            self.filterGeneMolsVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by molecules per gene. Min:" ,fg="black",bg="white").grid(column=0, row=10)
            tk.Entry(self.fileInfo, textvariable=self.filterGeneMolsVar).grid(column=1,row=10)

            #normalize
            self.normalizeVar = tk.BooleanVar()
            self.normalizeVar.set(True)
            tk.Checkbutton(self.fileInfo, text=u"Normalize by library size", variable=self.normalizeVar).grid(column=0, row=11, columnspan=4)

            #log transform
            self.logTransform = tk.BooleanVar()
            self.logTransform.set(False)
            tk.Checkbutton(self.fileInfo, text=u"Log-transform data", variable=self.logTransform).grid(column=0, row=12)

            self.pseudocount = tk.DoubleVar()
            self.pseudocount.set(0.1)
            tk.Label(self.fileInfo, text=u"Pseudocount (for log-transform)", fg="black",bg="white").grid(column=1, row=12)
            tk.Entry(self.fileInfo, textvariable=self.pseudocount).grid(column=2, row=12)

            tk.Button(self.fileInfo, text="Cancel", command=self.fileInfo.destroy).grid(column=1, row=13)
            tk.Button(self.fileInfo, text="Load", command=partial(self.processData, file_type='csv')).grid(column=2, row=13)

            self.wait_window(self.fileInfo)

    def loadMTX(self):
        self.dataFileName = filedialog.askopenfilename(title='Load data file', initialdir='~/.magic/data')
        if(self.dataFileName != ""):
            #pop up data options menu
            self.fileInfo = tk.Toplevel()
            self.fileInfo.title("Data options")
            tk.Label(self.fileInfo, text=u"File name: ").grid(column=0, row=0)
            tk.Label(self.fileInfo, text=self.dataFileName.split('/')[-1]).grid(column=1, row=0)

            tk.Label(self.fileInfo,text=u"Name:" ,fg="black",bg="white").grid(column=0, row=1)
            self.fileNameEntryVar = tk.StringVar()
            self.fileNameEntryVar.set('Data ' + str(len(self.data)))
            tk.Entry(self.fileInfo, textvariable=self.fileNameEntryVar).grid(column=1,row=1)

            tk.Button(self.fileInfo, text="Select gene names file", command=self.getGeneNameFile).grid(column=0, row=2)

            tk.Button(self.fileInfo, text="Compute data statistics", command=partial(self.showRawDataDistributions, file_type='mtx')).grid(column=0, row=3)

            #filter parameters
            self.filterCellMinVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by molecules per cell. Min:" ,fg="black",bg="white").grid(column=0, row=4)
            tk.Entry(self.fileInfo, textvariable=self.filterCellMinVar).grid(column=1,row=4)
            
            self.filterCellMaxVar = tk.StringVar()
            tk.Label(self.fileInfo, text=u" Max:" ,fg="black",bg="white").grid(column=2, row=4)
            tk.Entry(self.fileInfo, textvariable=self.filterCellMaxVar).grid(column=3,row=4)
            
            self.filterGeneNonzeroVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by nonzero cells per gene. Min:" ,fg="black",bg="white").grid(column=0, row=5)
            tk.Entry(self.fileInfo, textvariable=self.filterGeneNonzeroVar).grid(column=1,row=5)
            
            self.filterGeneMolsVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by molecules per gene. Min:" ,fg="black",bg="white").grid(column=0, row=6)
            tk.Entry(self.fileInfo, textvariable=self.filterGeneMolsVar).grid(column=1,row=6)

            #normalize
            self.normalizeVar = tk.BooleanVar()
            self.normalizeVar.set(True)
            tk.Checkbutton(self.fileInfo, text=u"Normalize by library size", variable=self.normalizeVar).grid(column=0, row=7, columnspan=4)

            #log transform
            self.logTransform = tk.BooleanVar()
            self.logTransform.set(False)
            tk.Checkbutton(self.fileInfo, text=u"Log-transform data", variable=self.logTransform).grid(column=0, row=8)

            self.pseudocount = tk.DoubleVar()
            self.pseudocount.set(0.1)
            tk.Label(self.fileInfo, text=u"Pseudocount (for log-transform)", fg="black",bg="white").grid(column=1, row=8)
            tk.Entry(self.fileInfo, textvariable=self.pseudocount).grid(column=2, row=8)

            tk.Button(self.fileInfo, text="Cancel", command=self.fileInfo.destroy).grid(column=1, row=10)
            tk.Button(self.fileInfo, text="Load", command=partial(self.processData, file_type='mtx')).grid(column=2, row=10)

            self.wait_window(self.fileInfo)

    def load10x(self):
        self.dataDir = filedialog.askdirectory(title='Select data directory', initialdir='~/.magic/data')
        if(self.dataDir != None):
            #pop up data options menu
            self.fileInfo = tk.Toplevel()
            self.fileInfo.title("Data options")
            tk.Label(self.fileInfo, text=u"Data directory: ").grid(column=0, row=0)
            tk.Label(self.fileInfo, text=self.dataDir).grid(column=1, row=0)

            tk.Label(self.fileInfo,text=u"Name:" ,fg="black",bg="white").grid(column=0, row=1)
            self.fileNameEntryVar = tk.StringVar()
            self.fileNameEntryVar.set('Data ' + str(len(self.data)))
            tk.Entry(self.fileInfo, textvariable=self.fileNameEntryVar).grid(column=1,row=1)

            tk.Label(self.fileInfo, text=u"Gene names:").grid(column=0, row=2)
            self.geneVar = tk.IntVar()
            self.geneVar.set(0)
            tk.Radiobutton(self.fileInfo, text='Use ensemble IDs', variable=self.geneVar, value=1).grid(column=1, row=2)
            tk.Radiobutton(self.fileInfo, text='Use gene names', variable=self.geneVar, value=0).grid(column=2, row=2)

            tk.Button(self.fileInfo, text="Compute data statistics", command=partial(self.showRawDataDistributions, file_type='10x')).grid(column=0, row=3)

            #filter parameters
            self.filterCellMinVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by molecules per cell. Min:" ,fg="black",bg="white").grid(column=0, row=4)
            tk.Entry(self.fileInfo, textvariable=self.filterCellMinVar).grid(column=1,row=4)
            
            self.filterCellMaxVar = tk.StringVar()
            tk.Label(self.fileInfo, text=u" Max:" ,fg="black",bg="white").grid(column=2, row=4)
            tk.Entry(self.fileInfo, textvariable=self.filterCellMaxVar).grid(column=3,row=4)
            
            self.filterGeneNonzeroVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by nonzero cells per gene. Min:" ,fg="black",bg="white").grid(column=0, row=5)
            tk.Entry(self.fileInfo, textvariable=self.filterGeneNonzeroVar).grid(column=1,row=5)
            
            self.filterGeneMolsVar = tk.StringVar()
            tk.Label(self.fileInfo,text=u"Filter by molecules per gene. Min:" ,fg="black",bg="white").grid(column=0, row=6)
            tk.Entry(self.fileInfo, textvariable=self.filterGeneMolsVar).grid(column=1,row=6)

            #normalize
            self.normalizeVar = tk.BooleanVar()
            self.normalizeVar.set(True)
            tk.Checkbutton(self.fileInfo, text=u"Normalize by library size", variable=self.normalizeVar).grid(column=0, row=7, columnspan=4)

            #log transform
            self.logTransform = tk.BooleanVar()
            self.logTransform.set(False)
            tk.Checkbutton(self.fileInfo, text=u"Log-transform data", variable=self.logTransform).grid(column=0, row=8)

            self.pseudocount = tk.DoubleVar()
            self.pseudocount.set(0.1)
            tk.Label(self.fileInfo, text=u"Pseudocount (for log-transform)", fg="black",bg="white").grid(column=1, row=8)
            tk.Entry(self.fileInfo, textvariable=self.pseudocount).grid(column=2, row=8)

            tk.Button(self.fileInfo, text="Cancel", command=self.fileInfo.destroy).grid(column=1, row=10)
            tk.Button(self.fileInfo, text="Load", command=partial(self.processData, file_type='10x')).grid(column=2, row=10)

            self.wait_window(self.fileInfo)

    def getGeneNameFile(self):
        self.geneNameFile = filedialog.askopenfilename(title='Select gene name file', initialdir='~/.magic/data')
        tk.Label(self.fileInfo,text=self.geneNameFile.split('/')[-1] ,fg="black",bg="white").grid(column=1, row=2)

    def loadPickle(self):
        self.dataFileName = filedialog.askopenfilename(title='Load saved session', initialdir='~/.magic/data')
        if(self.dataFileName != ""):
            #pop up data options menu
            self.fileInfo = tk.Toplevel()
            self.fileInfo.title("Data options")
            tk.Label(self.fileInfo, text=u"File name: ").grid(column=0, row=0)
            tk.Label(self.fileInfo, text=self.dataFileName.split('/')[-1]).grid(column=1, row=0)

            tk.Label(self.fileInfo,text=u"Name:" ,fg="black",bg="white").grid(column=0, row=1)
            self.fileNameEntryVar = tk.StringVar()
            self.fileNameEntryVar.set('Data ' + str(len(self.data)))
            tk.Entry(self.fileInfo, textvariable=self.fileNameEntryVar).grid(column=1,row=1)

            tk.Button(self.fileInfo, text="Cancel", command=self.fileInfo.destroy).grid(column=1, row=2)
            tk.Button(self.fileInfo, text="Load", command=partial(self.processData, file_type='pickle')).grid(column=2, row=2)

            self.wait_window(self.fileInfo)

    def processData(self, file_type='csv'):

        if len(self.data) == 0:
            #clear intro screen
            for item in self.grid_slaves():
                item.grid_forget()

            self.data_list = ttk.Treeview()
            self.data_list.heading('#0', text='Data sets')
            self.data_list.grid(column=0, row=0, rowspan=6, sticky='NSEW')
            self.data_list.bind('<BackSpace>', self._deleteDataItem)
            self.data_list.bind('<<TreeviewSelect>>', self._updateSelection)

            #make Treeview scrollable


            ysb = ttk.Scrollbar(orient=tk.VERTICAL, command=self.data_list.yview)
            xsb = ttk.Scrollbar(orient=tk.HORIZONTAL, command=self.data_list.xview)
            self.data_list.configure(yscroll=ysb.set, xscroll=xsb.set)

            self.data_detail = ttk.Treeview()
            self.data_detail.heading('#0', text='Features')
            self.data_detail.grid(column=0, row=5, rowspan=7, sticky='NSEW')

            ysb2 = ttk.Scrollbar(orient=tk.VERTICAL, command=self.data_detail.yview)
            xsb2 = ttk.Scrollbar(orient=tk.HORIZONTAL, command=self.data_detail.xview)
            self.data_detail.configure(yscroll=ysb2.set, xscroll=xsb2.set)

            self.notebook = ttk.Notebook(height=600, width=600)
            self.notebook.grid(column=1, row=0, rowspan=12, columnspan=4, sticky='NSEW')
            self.tabs = []

        if file_type == 'csv':    # sc-seq data
            scdata = magic.mg.SCData.from_csv(os.path.expanduser(self.dataFileName), data_type='sc-seq',
                                              cell_axis=self.rowVar.get(), delimiter=self.delimiter.get(),
                                              rows_after_header_to_skip=self.rowHeader.get(), 
                                              cols_after_header_to_skip=self.colHeader.get(),
                                              normalize=False)
        elif file_type == 'mtx':   # sparse matrix
            scdata = magic.mg.SCData.from_mtx(os.path.expanduser(self.dataFileName), 
                                              os.path.expanduser(self.geneNameFile),
                                              normalize=False)
        elif file_type == '10x':
            scdata = magic.mg.SCData.from_10x(self.dataDir, use_ensemble_id=self.geneVar.get(),
                                              normalize=False)
            
        if file_type != 'pickle':
            if len(self.filterCellMinVar.get()) > 0 or len(self.filterCellMaxVar.get()) > 0 or len(self.filterGeneNonzeroVar.get()) > 0 or len(self.filterGeneMolsVar.get()) > 0:
                scdata.filter_scseq_data(filter_cell_min=int(self.filterCellMinVar.get()) if len(self.filterCellMinVar.get()) > 0 else 0, 
                                         filter_cell_max=int(self.filterCellMaxVar.get()) if len(self.filterCellMaxVar.get()) > 0 else 0, 
                                         filter_gene_nonzero=int(self.filterGeneNonzeroVar.get()) if len(self.filterGeneNonzeroVar.get()) > 0 else 0, 
                                         filter_gene_mols=int(self.filterGeneMolsVar.get()) if len(self.filterGeneMolsVar.get()) > 0 else 0)

            if self.normalizeVar.get() == True:
                scdata = scdata.normalize_scseq_data() 

            if self.logTransform.get() == True:
                scdata.log_transform_scseq_data(pseudocount=self.pseudocount.get())

        else:   # pickled Wishbone object
            scdata = magic.mg.SCData.load(os.path.expanduser(self.dataFileName))

        self.data[self.fileNameEntryVar.get()] = {'scdata' : scdata, 'state' : tk.BooleanVar(),
                                                  'genes' : scdata.data.columns.values, 'gates' : {}}
        
        self.data_list.insert('', 'end', text=self.fileNameEntryVar.get() + ' (' + str(scdata.data.shape[0]) + ' x ' + str(scdata.data.shape[1]) + ')', open=True)

        #enable buttons
        self.analysisMenu.entryconfig(0, state='normal')
        self.analysisMenu.entryconfig(1, state='normal')
        self.analysisMenu.entryconfig(2, state='normal')
        self.fileMenu.entryconfig(4, state='normal')
        self.visMenu.entryconfig(0, state='normal')
        self.visMenu.entryconfig(1, state='normal')
        self.visMenu.entryconfig(2, state='normal')
        # self.concatButton = tk.Button(self, text=u"Concatenate selected datasets", state='disabled', wraplength=80, command=self.concatenateData)
        # self.concatButton.grid(column=0, row=11)
        if len(self.data) > 1:
            self.fileMenu.entryconfig(6, state='normal')

        self.geometry('1000x650')
        #destroy pop up menu
        self.fileInfo.destroy()

    def saveData(self):
        for key in self.data_list.selection():
            name = self.data_list.item(key)['text'].split(' (')[0]
            pickleFileName = filedialog.asksaveasfilename(title=name + ': save data', defaultextension='.p', initialfile=key)
            if pickleFileName != None:
                self.data[name]['scdata'].save(pickleFileName)

    def concatenateData(self):
        self.concatOptions = tk.Toplevel()
        self.concatOptions.title("Concatenate data sets")

        tk.Label(self.concatOptions, text=u"New data set name:", fg="black", bg="white").grid(column=0, row=0)
        self.nameVar = tk.StringVar()
        tk.Entry(self.concatOptions, textvariable=self.nameVar).grid(column=1, row=0)

        self.colVar = tk.IntVar()
        tk.Radiobutton(self.concatOptions, text='Concatenate columns', variable=self.colVar, value=0).grid(column=0, row=1)
        tk.Radiobutton(self.concatOptions, text='Concatenate rows', variable=self.colVar, value=1).grid(column=1, row=1)

        self.joinVar = tk.BooleanVar()
        self.joinVar.set(True)
        tk.Checkbutton(self.concatOptions, text=u"Outer join", variable=self.joinVar).grid(column=0, row=2, columnspan=2)

        tk.Button(self.concatOptions, text="Concatenate", command=self._concatenateData).grid(column=1, row=3)
        tk.Button(self.concatOptions, text="Cancel", command=self.concatOptions.destroy).grid(column=0, row=3)
        self.wait_window(self.concatOptions)

    def _concatenateData(self):
        to_concat = []
        for key in self.data_list.selection():
                to_concat.append(self.data[self.data_list.item(key)['text'].split(' (')[0]]['scdata'])

        scdata = to_concat[0].concatenate_data(to_concat[1:], axis=self.colVar.get(), join='outer' if self.joinVar.get() == True else 'inner',
                                               names=[self.data_list.item(key)['text'].split(' (')[0] for key in self.data_list.selection()])


        self.data[self.nameVar.get()] = {'scdata' : scdata, 'wb' : None, 'state' : tk.BooleanVar(),
                                         'genes' : scdata.data.columns.values, 'gates' : {}}
        self.data_list.insert('', 'end', text=self.nameVar.get() + ' (' + str(scdata.data.shape[0]) + ' x ' + str(scdata.data.shape[1]) + ')', open=True)

        self.concatOptions.destroy()

    def _deleteDataItem(self, event):

        self.data_detail.delete(*self.data_detail.get_children())
        for key in self.data_list.selection():
            name = self.data_list.item(key)['text'].split(' (')[0]
            if name in self.data:
                del self.data[name]
            else:
                data_set_name = self.data_list.item(self.data_list.parent(key))['text'].split(' (')[0]
                if 'Principal components' in name:
                    self.data[data_set_name]['scdata'].pca = None
                elif 'tSNE' in name:
                    self.data[data_set_name]['scdata'].tsne = None
                elif 'Diffusion components' in name:
                    self.data[data_set_name]['scdata'].diffusion_eigenvectors = None
                    self.data[data_set_name]['scdata'].diffusion_eigenvalues = None
                elif 'Wishbone' in name:
                    del self.data[data_set_name]['wb']
                elif 'MAGIC' in name:
                    del self.data[data_set_name + ' MAGIC']
                    self.data[data_set_name]['scdata'].magic = None
            self.data_list.delete(key)

    def _updateSelection(self, event):

        self.data_detail.delete(*self.data_detail.get_children())

        for key in self.data_list.selection():
            name = self.data_list.item(key)['text'].split(' (')[0]

            magic = False
            if 'MAGIC' in name:
                magic = True

            if 'PCA' in name:
                data_set = name.split(' PCA')[0]
                for i in range(self.data[data_set]['scdata'].pca.shape[1]):
                    if magic:
                        self.data_detail.insert('', 'end', text='MAGIC PC'+str(i+1), open=True)
                    else:
                        self.data_detail.insert('', 'end', text='PC'+str(i+1), open=True)

            elif 'tSNE' in name:
                data_set = name.split(' tSNE')[0]
                for i in range(self.data[data_set]['scdata'].tsne.shape[1]):
                    if magic:
                        self.data_detail.insert('', 'end', text='MAGIC tSNE'+str(i+1), open=True)
                    else:
                        self.data_detail.insert('', 'end', text='tSNE'+str(i+1), open=True)

            elif 'Diffusion components' in name:
                data_set = name.split(' Diffusion components')[0]
                for i in range(len(self.data[data_set]['scdata'].diffusion_eigenvectors)):
                    if magic:
                        self.data_detail.insert('', 'end', text='MAGIC DC'+str(i+1), open=True)
                    else:
                        self.data_detail.insert('', 'end', text='DC'+str(i+1), open=True)

            else:
                for gene in self.data[name]['scdata'].data:
                    if magic:
                        self.data_detail.insert('', 'end', text='MAGIC ' + gene, open=True)
                    else:
                        self.data_detail.insert('', 'end', text=gene, open=True)

    def showRawDataDistributions(self, file_type='csv'):
        if file_type == 'csv':    # sc-seq data
            scdata = magic.mg.SCData.from_csv(os.path.expanduser(self.dataFileName), 
                                              data_type='sc-seq', normalize=False)
        elif file_type == 'mtx':   # sparse matrix
            scdata = magic.mg.SCData.from_mtx(os.path.expanduser(self.dataFileName), 
                                              os.path.expanduser(self.geneNameFile))
        elif file_type == '10x':
            scdata = magic.mg.SCData.from_10x(self.dataDir)

        self.dataDistributions = tk.Toplevel()
        self.dataDistributions.title(self.fileNameEntryVar.get() + ": raw data distributions")

        fig, ax = scdata.plot_molecules_per_cell_and_gene()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.dataDistributions)
        canvas.show()
        canvas.get_tk_widget().grid(column=0, row=0, rowspan=10, columnspan=4,sticky='NSEW')
        del scdata

        self.wait_window(self.dataDistributions)

    def runPCA(self):
        self.pcaOptions = tk.Toplevel()
        self.pcaOptions.title("PCA options")

        tk.Label(self.pcaOptions, text=u"Number of components:", fg="black",bg="white").grid(column=0, row=0)
        self.nComponents = tk.IntVar()
        self.nComponents.set(20)
        tk.Entry(self.pcaOptions, textvariable=self.nComponents).grid(column=1, row=0)

        self.randomVar = tk.BooleanVar()
        self.randomVar.set(True)
        tk.Checkbutton(self.pcaOptions, text=u"Randomized PCA (faster)", variable=self.randomVar).grid(column=0, row=2, columnspan=2)

        tk.Button(self.pcaOptions, text="Run", command=self._runPCA).grid(column=1, row=4)
        tk.Button(self.pcaOptions, text="Cancel", command=self.pcaOptions.destroy).grid(column=0, row=4)
        self.wait_window(self.pcaOptions)

    def _runPCA(self):
        for key in self.data_list.selection():
            curKey = self.data_list.item(key)['text'].split(' (')[0]

            # run PCA on the selected dataset
            if self.data[curKey]['scdata'].pca is None:
                self.data[curKey]['scdata'].run_pca(n_components=self.nComponents.get(), random=self.randomVar.get())

            self.data_list.insert(key, 'end', text=curKey + ' PCA' + 
                              ' (' + str(self.data[curKey]['scdata'].pca.shape[0]) + 
                                ' x ' + str(self.data[curKey]['scdata'].pca.shape[1]) + ')', open=True)

            # plot component-variance plot with the input number of components
            self.fig = plt.figure(figsize=[6, 6])
            self.fig, self.ax = self.data[curKey]['scdata'].plot_pca_variance_explained(n_components=self.nComponents.get(),
                                                                                        fig=self.fig,
                                                                                        random=self.randomVar.get())

            self.tabs.append([tk.Frame(self.notebook), self.fig])
            self.notebook.add(self.tabs[len(self.tabs) - 1][0], text='PCA plot')

            self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs) - 1][0])
            self.canvas.show()
            self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

            self.fileMenu.entryconfig(5, state='normal')

            self.currentPlot = 'pca'
            self.pcaOptions.destroy()

    def runMagic(self):
        for key in self.data_list.selection():
            # pop up for parameters
            self.magicOptions = tk.Toplevel()
            self.magicOptions.title(self.data_list.item(key)['text'].split(' (')[0] + ": MAGIC options")
            self.curKey = key

            tk.Label(self.magicOptions, text=u"# of PCA components:", fg="black", bg="white").grid(column=0, row=1)
            self.nCompVar = tk.IntVar()
            self.nCompVar.set(20)
            tk.Entry(self.magicOptions, textvariable=self.nCompVar).grid(column=1, row=1)

            self.randomVar = tk.BooleanVar()
            self.randomVar.set(True)
            tk.Checkbutton(self.magicOptions, text=u"Randomized PCA", variable=self.randomVar).grid(column=0, row=2,
                                                                                                    columnspan=2)

            tk.Label(self.magicOptions, text=u"t:", fg="black", bg="white").grid(column=0, row=3)
            self.tVar = tk.IntVar()
            self.tVar.set(6)
            tk.Entry(self.magicOptions, textvariable=self.tVar).grid(column=1, row=3)

            tk.Label(self.magicOptions, text=u"k:", fg="black", bg="white").grid(column=0, row=4)
            self.kVar = tk.IntVar()
            self.kVar.set(30)
            tk.Entry(self.magicOptions, textvariable=self.kVar).grid(column=1, row=4)

            tk.Label(self.magicOptions, text=u"ka:", fg="black", bg="white").grid(column=0, row=5)
            self.autotuneVar = tk.IntVar()
            self.autotuneVar.set(10)
            tk.Entry(self.magicOptions, textvariable=self.autotuneVar).grid(column=1, row=5)

            tk.Label(self.magicOptions, text=u"Epsilon:", fg="black", bg="white").grid(column=0, row=6)
            self.epsilonVar = tk.IntVar()
            self.epsilonVar.set(1)
            tk.Entry(self.magicOptions, textvariable=self.epsilonVar).grid(column=1, row=6)

            tk.Label(self.magicOptions, text=u"(Epsilon 0 is the uniform kernel)", fg="black", bg="white").grid(
                column=0, columnspan=2, row=7)

            self.rescaleVar = tk.IntVar()
            self.rescaleVar.set(99)
            tk.Label(self.magicOptions, text=u"Rescale data to ", fg="black", bg="white").grid(column=0, row=8)
            tk.Entry(self.magicOptions, textvariable=self.rescaleVar).grid(column=1, row=8)
            tk.Label(self.magicOptions, text=u" percentile", fg="black", bg="white").grid(column=2, row=8)
            tk.Label(self.magicOptions, text=u"0 is no rescale (use for log-transformed data).").grid(row=9, column=0,
                                                                                                      columnspan=2)

            tk.Button(self.magicOptions, text="Cancel", command=self.magicOptions.destroy).grid(column=0, row=10)
            tk.Button(self.magicOptions, text="Run", command=self._runMagic).grid(column=1, row=10)
            self.wait_window(self.magicOptions)

    def _runMagic(self):
        name = self.data_list.item(self.curKey)['text'].split(' (')[0]

        self.magicOptions.destroy()
        self.magicProgress = tk.Toplevel()
        self.magicProgress.title(name + ': Running MAGIC')
        tk.Label(self.magicProgress, text="Running MAGIC - refer to console for progress updates.").grid(column=0,
                                                                                                         row=0)
        self.magicProgress.update()

        if self.data[name]['scdata'].magic is None:
            self.data[name]['scdata'].run_magic(n_pca_components=self.nCompVar.get() if self.nCompVar.get() > 0 else None,
                                            t=self.tVar.get(), k=self.kVar.get(), epsilon=self.epsilonVar.get(),
                                            rescale_percent=self.rescaleVar.get(), ka=self.autotuneVar.get(),
                                            random_pca=self.randomVar.get())

        self.data[name + ' MAGIC'] = {'scdata': self.data[name]['scdata'].magic, 'wb': None, 'state': tk.BooleanVar(),
                                      'genes': self.data[name]['scdata'].magic.data.columns.values, 'gates': {}}

        self.data_list.insert(self.curKey, 'end', text=name + ' MAGIC' +
                                                       ' (' + str(self.data[name]['scdata'].magic.data.shape[0]) +
                                                       ' x ' + str(self.data[name]['scdata'].magic.data.shape[1]) + ')',
                              open=True)

        self.magicProgress.destroy()

    def runPhenoGraph(self):
        for key in self.data_list.selection():
            # pop up for parameters
            self.phenoOptions = tk.Toplevel()
            self.phenoOptions.title(self.data_list.item(key)['text'].split(' (')[0] + ": PhenoGraph options")
            self.curKey = key

            tk.Label(self.phenoOptions, text=u"# of Nearest Neighbors:", fg="black", bg="white").grid(column=0, row=1)
            self.nnnumVar = tk.IntVar()
            self.nnnumVar.set(30)
            tk.Entry(self.phenoOptions, textvariable=self.nnnumVar).grid(column=1, row=1)

            tk.Label(self.phenoOptions, text=u"Minimum cluster size:", fg="black", bg="white").grid(column=0, row=2)
            self.minsVar = tk.IntVar()
            self.minsVar.set(10)
            tk.Entry(self.phenoOptions, textvariable=self.minsVar).grid(column=1, row=2)

            tk.Label(self.phenoOptions, text=u"Distance metric:", fg="black", bg="white").grid(column=0, row=3)
            self.choiceVar = tk.StringVar()
            choices = {'euclidean', 'manhattan', 'correlation', 'cosine'}
            self.choiceVar.set('euclidean')
            tk.OptionMenu(self.phenoOptions, self.choiceVar, *choices).grid(column=1, row=3)

            tk.Label(self.phenoOptions, text=u"Number of jobs:", fg="black", bg="white").grid(column=0, row=4)
            self.njobVar = tk.IntVar()
            self.njobVar.set(-1)
            tk.Entry(self.phenoOptions, textvariable=self.njobVar).grid(column=1, row=4)

            tk.Label(self.phenoOptions, text=u"Tolerance:", fg="black", bg="white").grid(column=0, row=5)
            self.toleVar = tk.DoubleVar()
            self.toleVar.set(1e-3)
            tk.Entry(self.phenoOptions, textvariable=self.toleVar).grid(column=1, row=5)

            tk.Label(self.phenoOptions, text=u"Louvain time limit (s):", fg="black", bg="white").grid(column=0, row=6)
            self.timeVar = tk.IntVar()
            self.timeVar.set(2000)
            tk.Entry(self.phenoOptions, textvariable=self.timeVar).grid(column=1, row=6)

            tk.Label(self.phenoOptions, text=u"nn-method:", fg="black", bg="white").grid(column=0, row=7)
            self.nnVar = tk.StringVar()
            nn_choices = {'brute force', 'kdtree'}
            self.nnVar.set('kdtree')
            tk.OptionMenu(self.phenoOptions, self.nnVar, *nn_choices).grid(column=1, row=7)

            self.directedVar = tk.BooleanVar()
            self.directedVar.set(False)
            tk.Checkbutton(self.phenoOptions, text=u"Directed Graph", variable=self.directedVar).grid(column=0, row=8)

            self.pruneVar = tk.BooleanVar()
            self.pruneVar.set(False)
            tk.Checkbutton(self.phenoOptions, text=u"Prune", variable=self.pruneVar).grid(column=1, row=8)

            self.jaccVar = tk.BooleanVar()
            self.jaccVar.set(True)
            tk.Checkbutton(self.phenoOptions, text=u"Jaccard Metric", variable=self.jaccVar).grid(column=2, row=8,
                                                                                                  columnspan=2)

            tk.Button(self.phenoOptions, text="Cancel", command=self.phenoOptions.destroy).grid(column=0, row=9)
            tk.Button(self.phenoOptions, text="Run", command=self._runPhenoGraph).grid(column=1, row=9)
            self.wait_window(self.phenoOptions)

    def _runPhenoGraph(self):
        name = self.data_list.item(self.curKey)['text'].split(' (')[0]
        scdata = self.data[name]['scdata']

        self.phenoOptions.destroy()
        self.phenoProgress = tk.Toplevel()
        self.msgVar = tk.StringVar()
        self.msgVar.set("starting PhenoGraph...")
        self.phenoProgress.title(name + ': Running PhenoGraph')
        self.msgLb = tk.Label(self.phenoProgress, textvariable=self.msgVar).pack()

        self.phenoProgress.update()

        print("log-transforming the data...")
        self.msgVar.set("log-transforming the data...")
        self.phenoProgress.update()
        scdata.log_transform_scseq_data()

        print("running tSNE...")
        self.msgVar.set("running tSNE...")
        self.phenoProgress.update()
        if scdata.tsne is None:
            scdata.run_tsne()

        print("running PhenoGraph...")
        self.msgVar.set("running PhenoGraph...")
        self.phenoProgress.update()
        communities, graph, Q = phenograph.cluster(scdata.data, k=self.nnnumVar.get(),
                                                   directed=self.directedVar.get(), prune=self.pruneVar.get(),
                                                   min_cluster_size=self.minsVar.get(), jaccard=self.jaccVar.get(),
                                                   primary_metric=self.choiceVar.get(), n_jobs=self.njobVar.get(),
                                                   q_tol=self.toleVar.get(), louvain_time_limit=self.timeVar.get(),
                                                   nn_method=self.nnVar.get())
        numCluster = np.max(communities)+1
        communities = [str(int(i)+1) for i in communities]


        print("plotting data points...")
        self.msgVar.set("plotting data points...")
        self.phenoProgress.update()
        self.plotPheno(scdata, pd.Series(communities))
        self.phenoProgress.destroy()

        self.phenoResult = tk.Toplevel()
        self.phenoResult.title(name + " PhenoGraph Results")
        tk.Label(self.phenoResult, text=u"# of Clusters: " + str(numCluster), fg="black", bg="white").grid(column=0, row=1)
        tk.Label(self.phenoResult, text=u"Modularity Score: " + str(Q), fg="black", bg="white").grid(column=0, row=2)
        tk.Button(self.phenoResult, text="Ok", command=self.phenoResult.destroy).grid(column=0, row=3)
        # to be implemented
        tk.Button(self.phenoResult, text="Save Communities as CSV",
                  command=lambda: self.saveCSV(scdata, pd.Series(communities))).grid(column=1, row=3)
        self.phenoResult.update()
        self.wait_window(self.phenoResult)

    def runTSNE(self):
        for key in self.data_list.selection():
            #pop up for # components
            self.tsneOptions = tk.Toplevel()
            self.curKey = key
            name = self.data_list.item(key)['text'].split(' (')[0]
            self.tsneOptions.title(name + ": tSNE options")
            if self.data[name]['scdata'].data_type == 'sc-seq':
                tk.Label(self.tsneOptions,text=u"Number of components:" ,fg="black",bg="white").grid(column=0, row=0)
                self.nCompVar = tk.IntVar()
                self.nCompVar.set(50)
                tk.Entry(self.tsneOptions, textvariable=self.nCompVar).grid(column=1,row=0)

            tk.Label(self.tsneOptions,text=u"Perplexity:" ,fg="black",bg="white").grid(column=0, row=1)
            self.perplexityVar = tk.IntVar()
            self.perplexityVar.set(30)
            tk.Entry(self.tsneOptions, textvariable=self.perplexityVar).grid(column=1,row=1)

            tk.Label(self.tsneOptions,text=u"Number of iterations:" ,fg="black",bg="white").grid(column=0, row=2)
            self.iterVar = tk.IntVar()
            self.iterVar.set(1000)
            tk.Entry(self.tsneOptions, textvariable=self.iterVar).grid(column=1,row=2)

            tk.Label(self.tsneOptions,text=u"Theta:" ,fg="black",bg="white").grid(column=0, row=3)
            self.angleVar = tk.DoubleVar()
            self.angleVar.set(0.5)
            tk.Entry(self.tsneOptions, textvariable=self.angleVar).grid(column=1,row=3)

            tk.Button(self.tsneOptions, text="Run", command=self._runTSNE).grid(column=1, row=4)
            tk.Button(self.tsneOptions, text="Cancel", command=self.tsneOptions.destroy).grid(column=0, row=4)
            self.wait_window(self.tsneOptions)

    def _runTSNE(self):
        name = self.data_list.item(self.curKey)['text'].split(' (')[0]
        if self.data[name]['scdata'].tsne is None:
            self.data[name]['scdata'].run_tsne(n_components=self.nCompVar.get(), perplexity=self.perplexityVar.get(),
                                           n_iter=self.iterVar.get(), theta=self.angleVar.get())
        self.data[name]['gates'] = {}

        #enable buttons
        self.analysisMenu.entryconfig(2, state='normal')

        self.data_list.insert(self.curKey, 'end', text=name + ' tSNE' + 
                              ' (' + str(self.data[name]['scdata'].tsne.shape[0]) + 
                                ' x ' + str(self.data[name]['scdata'].tsne.shape[1]) + ')', open=True)
        self.plotTSNE()
        self.tsneOptions.destroy()

    def runDM(self):
        for key in self.data_list.selection():
            #pop up for # components
            self.DMOptions = tk.Toplevel()
            self.curKey = key
            name = self.data_list.item(key)['text'].split(' (')[0]
            self.DMOptions.title(name + ": Diffusion map options")

            tk.Label(self.DMOptions,text=u"Number of components:" ,fg="black",bg="white").grid(column=0, row=0)
            self.nCompVar = tk.IntVar()
            self.nCompVar.set(10)
            tk.Entry(self.DMOptions, textvariable=self.nCompVar).grid(column=1,row=0)

            tk.Label(self.DMOptions,text=u"Number of PCA components:" ,fg="black",bg="white").grid(column=0, row=1)
            self.nPCAVar = tk.IntVar()
            self.nPCAVar.set(20)
            tk.Entry(self.DMOptions, textvariable=self.nPCAVar).grid(column=1,row=1)

            self.randomPCAVar = tk.BooleanVar()
            self.randomPCAVar.set(True)
            tk.Checkbutton(self.DMOptions, text=u"Randomized PCA (faster)", variable=self.randomPCAVar).grid(column=0, row=2, columnspan=2)

            tk.Label(self.DMOptions,text=u"k:" ,fg="black",bg="white").grid(column=0, row=3)
            self.kVar = tk.IntVar()
            self.kVar.set(30)
            tk.Entry(self.DMOptions, textvariable=self.kVar).grid(column=1,row=3)

            tk.Label(self.DMOptions,text=u"ka:" ,fg="black",bg="white").grid(column=0, row=4)
            self.autotuneVar = tk.IntVar()
            self.autotuneVar.set(10)
            tk.Entry(self.DMOptions, textvariable=self.autotuneVar).grid(column=1,row=4)

            tk.Label(self.DMOptions,text=u"Epsilon:" ,fg="black",bg="white").grid(column=0, row=5)
            self.epsilonVar = tk.IntVar()
            self.epsilonVar.set(1)
            tk.Entry(self.DMOptions, textvariable=self.epsilonVar).grid(column=1,row=5)

            tk.Label(self.DMOptions, text=u"(Epsilon 0 is the uniform kernel)",fg="black",bg="white").grid(column=0,columnspan=2,row=6)

            tk.Button(self.DMOptions, text="Run", command=self._runDM).grid(column=1, row=7)
            tk.Button(self.DMOptions, text="Cancel", command=self.DMOptions.destroy).grid(column=0, row=7)
            self.wait_window(self.DMOptions)

    def _runDM(self):
        for key in self.data_list.selection():
            name = self.data_list.item(key)['text'].split(' (')[0]
            if self.data[name]['scdata'].diffusion_eigenvectors is None:
                self.data[name]['scdata'].run_diffusion_map(n_diffusion_components=self.nCompVar.get(), epsilon=self.epsilonVar.get(), n_pca_components=self.nPCAVar.get(),
                                                        k=self.kVar.get(), ka=self.autotuneVar.get(), random_pca=self.randomPCAVar.get())

            self.data_list.insert(key, 'end', text=name + ' Diffusion components' +
                                  ' (' + str(self.data[name]['scdata'].diffusion_eigenvectors.shape[0]) + 
                                 ' x ' + str(self.data[name]['scdata'].diffusion_eigenvectors.shape[1]) + ')', open=True)
            self.DMOptions.destroy()

    # already integrated into the PCA function
    def plotPCAVariance(self):
        for key in self.data_list.selection():
            #pop up for parameters
            self.plotOptions = tk.Toplevel()
            self.plotOptions.title(self.data_list.item(key)['text'].split(' (')[0] + ": PCA plot options")
            self.curKey = key

            tk.Label(self.plotOptions,text=u"# of PCA components:" ,fg="black",bg="white").grid(column=0, row=1)
            self.nCompVar = tk.IntVar()
            self.nCompVar.set(40)
            tk.Entry(self.plotOptions, textvariable=self.nCompVar).grid(column=1,row=1)

            self.randomVar = tk.BooleanVar()
            self.randomVar.set(True)
            tk.Checkbutton(self.plotOptions, text=u"Randomized PCA", variable=self.randomVar).grid(column=0, row=2, columnspan=2)

            tk.Button(self.plotOptions, text="Cancel", command=self.plotOptions.destroy).grid(column=0, row=3)
            tk.Button(self.plotOptions, text="Run", command=self._plotPCAVariance).grid(column=1, row=3)
            self.wait_window(self.plotOptions)

    # already integrated into the PCA function
    def _plotPCAVariance(self):
        name = self.data_list.item(self.curKey)['text'].split(' (')[0]
        self.fig = plt.figure(figsize=[6,6])
        self.fig, self.ax = self.data[name]['scdata'].plot_pca_variance_explained(n_components=self.nCompVar.get(), fig=self.fig,
                                                                                  random=self.randomVar.get())

        self.tabs.append([tk.Frame(self.notebook), self.fig])
        self.notebook.add(self.tabs[len(self.tabs)-1][0], text='PCA plot')

        self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs)-1][0])
        self.canvas.show()
        self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

        tk.Button(self.tabs[len(self.tabs)-1][0], text="Save", command=self.savePlot).grid(row=0, column=5, sticky='NE')
        tk.Button(self.tabs[len(self.tabs)-1][0], text="Close tab", command=self.closeCurrentTab).grid(row=1, column=5, sticky='NE')
        self.currentPlot = 'pca'
        self.plotOptions.destroy()

    def plotPCA_DM(self):
        keys = self.data_list.selection()
        name = self.data_list.item(keys[0])['text'].split(' (')[0]
        plot_type = ''
        if 'PCA' in name:
            plot_type='PCA'
        elif 'Diffusion components' in name:
            plot_type='Diffusion components'
        name = name.split(' ' + plot_type)[0]

        self.getScatterSelection(plot_type=plot_type, 
            options=self.data[name]['scdata'].pca.columns.values if plot_type == 'PCA' else self.data[name]['scdata'].diffusion_eigenvectors.columns.values)

        if self.zVar.get() == 'None':
            components = [self.xVar.get(), self.yVar.get()]
        else:
            components = [self.xVar.get(), self.yVar.get(), self.zVar.get()]

        colorSelection = self.colorVar.get().split(', ')
        if (len(colorSelection) == 1 and len(colorSelection[0]) > 0) or len(colorSelection) > 1:
            if len(colorSelection) == 1 and len(keys) == 1:
                self.fig = plt.figure(figsize=[6*len(colorSelection), 6 * len(keys)])
            else:
                self.fig = plt.figure(figsize=[4*len(colorSelection), 4 * len(keys)])

            gs = gridspec.GridSpec(len(keys), len(colorSelection))
            self.ax = []

            for i in range(len(keys)):

                name = self.data_list.item(keys[i])['text'].split(' (')[0]
                if plot_type in name:
                    name = name.split(' ' + plot_type)[0]

                for j in range(len(colorSelection)):

                    if len(components) == 3:
                        self.ax.append(self.fig.add_subplot(gs[i, j], projection='3d'))
                    else:
                        self.ax.append(self.fig.add_subplot(gs[i, j]))

                    if colorSelection[j] in self.data[name]['scdata'].extended_data.columns.get_level_values(1):
                        self.data[name]['scdata'].scatter_gene_expression(components,fig=self.fig, 
                                                                          ax=self.ax[len(self.ax)-1], color=colorSelection[j])
                    elif colorSelection[j] == 'density':
                        self.data[name]['scdata'].scatter_gene_expression(components, fig=self.fig, 
                                                                          ax=self.ax[len(self.ax)-1], density=True)
                    else:
                        self.data[name]['scdata'].scatter_gene_expression(components, fig=self.fig, 
                                                                          ax=self.ax[len(self.ax)-1], color=colorSelection[j])

                    self.ax[len(self.ax)-1].set_title(name + ' (color =' + colorSelection[j] + ')')
                    self.ax[len(self.ax)-1].set_xlabel(' '.join(components[0]))
                    self.ax[len(self.ax)-1].set_ylabel(' '.join(components[1]))
                    if len(components) == 3:
                        self.ax[len(self.ax)-1].set_zlabel(' '.join(components[2]))

            gs.tight_layout(self.fig)

            self.tabs.append([tk.Frame(self.notebook), self.fig])
            self.notebook.add(self.tabs[len(self.tabs)-1][0], text=self.plotNameVar.get())

            self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs)-1][0])
            self.canvas.show()
            self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

            self.fileMenu.entryconfig(5, state='normal')

            if len(components) == 3:
                for ax in self.ax:
                    ax.mouse_init()

            self.currentPlot = plot_type

    def plotTSNE(self):
        keys = self.data_list.selection()
        self.getScatterSelection(plot_type='tsne')

        self.colorSelection = self.colorVar.get().split(', ')
        if (len(self.colorSelection) == 1 and len(self.colorSelection[0]) > 0) or len(self.colorSelection) > 1:
            if len(self.colorSelection) == 1 and len(keys) == 1:
                self.fig = plt.figure(figsize=[6*len(self.colorSelection), 6 * len(keys)])
            else:
                self.fig = plt.figure(figsize=[4*len(self.colorSelection), 4 * len(keys)])
            gs = gridspec.GridSpec(len(keys), len(self.colorSelection))
            for i in range(len(keys)):
                name = self.data_list.item(keys[i])['text'].split(' (')[0]
                if 'tSNE' in name:
                    name = name.split(' tSNE')[0]
                for j in range(len(self.colorSelection)):
                    self.ax = self.fig.add_subplot(gs[i, j])
                    if 'PC' in self.colorSelection[j]:
                        self.fig, self.ax = self.data[name]['scdata'].plot_tsne(fig=self.fig, ax=self.ax,
                                                                                color=self.data[name]['scdata'].pca[self.colorSelection[j]])
                    elif 'DC' in self.colorSelection[j]:
                        self.fig, self.ax = self.data[name]['scdata'].plot_tsne(fig=self.fig, ax=self.ax, 
                                                                                color=self.data[name]['scdata'].diffusion_eigenvectors[self.colorSelection[j]])
                    elif 'MAGIC' in self.colorSelection[j]:
                        color = self.colorSelection[j].split('MAGIC ')[1]
                        self.fig, self.ax = self.data[name]['scdata'].plot_tsne(fig=self.fig, ax=self.ax, 
                                                                                color=self.data[name]['scdata'].magic.data[color])
                    elif self.colorSelection[j] in self.data[name]['scdata'].data.columns:
                        self.fig, self.ax = self.data[name]['scdata'].plot_tsne(fig=self.fig, ax=self.ax, 
                                                                                color=self.data[name]['scdata'].data[self.colorSelection[j]])
                    elif self.colorSelection[j] == 'density':
                        self.data[name]['scdata'].plot_tsne(fig=self.fig, ax=self.ax, density=True)
                    else:
                        self.data[name]['scdata'].plot_tsne(fig=self.fig, ax=self.ax, color=self.colorSelection[j])
                    self.ax.set_title(name + ' (color =' + self.colorSelection[j] + ')')
                    self.ax.set_xlabel('tSNE1')
                    self.ax.set_ylabel('tSNE2')
            gs.tight_layout(self.fig)

            self.tabs.append([tk.Frame(self.notebook), self.fig])
            self.notebook.add(self.tabs[len(self.tabs)-1][0], text=self.plotNameVar.get())

            self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs)-1][0])
            self.canvas.show()
            self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

            self.fileMenu.entryconfig(5, state='normal')

            self.currentPlot = 'tsne'

    def plotPheno(self, scdata, col):
        """
        :param scdata: SCdata object
        :param col: pd.Series object
        """
        name = self.data_list.item(self.curKey)['text'].split(' (')[0]
        toPlot = scdata.tsne.assign(com=pd.Series(col).values)
        clusterRec = {}

        self.fig = plt.figure(figsize=[6, 6])
        gs = gridspec.GridSpec(1, 1)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.fig, self.ax = scdata.plot_tsne(fig=self.fig, ax=self.ax, color=col)
        self.ax.set_title('PhenoGraph Clustering Result')
        self.ax.set_xlabel('tSNE1')
        self.ax.set_ylabel('tSNE2')

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

        gs.tight_layout(self.fig)

        self.tabs.append([tk.Frame(self.notebook), self.fig])
        self.notebook.add(self.tabs[len(self.tabs) - 1][0], text="Phenograph")

        self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs) - 1][0])
        self.canvas.show()
        self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

        self.fileMenu.entryconfig(5, state='normal')

        self.currentPlot = 'tsne'

    def scatterPlot(self):
        keys = self.data_list.selection()
        if 'tSNE' in self.data_list.item(keys[0])['text']:
            self.plotTSNE()
        elif 'PCA' in self.data_list.item(keys[0])['text'] or 'Diffusion components' in self.data_list.item(keys[0])['text']:
            self.plotPCA_DM()
        else:
            self.getScatterSelection()
            xSelection = self.xVar.get().split(', ')
            ySelection = self.yVar.get().split(', ')
            zSelection = self.zVar.get().split(', ')
            colorSelection = self.colorVar.get().split(', ')
            if len(xSelection[0]) > 0 and len(ySelection[0]) > 0 and len(xSelection) == len(ySelection):
                if len(colorSelection) == 1 and len(colorSelection) != len(xSelection):
                    colorSelection = np.repeat(colorSelection, len(xSelection))

                keys = self.data_list.selection()
                if len(xSelection) == 1 and len(keys) == 1:
                    self.fig = plt.figure(figsize=[6*len(xSelection), 6 * len(keys)])
                else:
                    self.fig = plt.figure(figsize=[4*len(xSelection), 4*len(keys)])
                gs = gridspec.GridSpec(len(keys), len(xSelection))
                self.ax = []
                for i in range(len(keys)):
                    name = self.data_list.item(keys[i])['text'].split(' (')[0]

                    for j in range(len(xSelection)):

                        if len(zSelection[0]) > 0 and len(zSelection) == len(xSelection):
                            self.ax.append(self.fig.add_subplot(gs[i, j], projection='3d'))
                            genes = [xSelection[j], ySelection[j], zSelection[j]]
                        else:
                            self.ax.append(self.fig.add_subplot(gs[i, j]))
                            genes = [xSelection[j], ySelection[j]]

                        if colorSelection[j] in self.data[name]['scdata'].extended_data.columns.get_level_values(1):
                            self.data[name]['scdata'].scatter_gene_expression(genes, fig=self.fig, ax=self.ax[len(self.ax)-1],
                                                                                  color=colorSelection[j])
                        elif 'MAGIC' in colorSelection[j]:
                            color = colorSelection[j].split('MAGIC ')[0]
                            if color in self.data[name]['scdata'].magic.data.columns:
                                    self.data[name]['scdata'].scatter_gene_expression(genes, fig=self.fig, ax=self.ax[len(self.ax)-1],
                                                                                      color=self.data[name]['scdata'].magic.data[color])
                            
                        elif colorSelection[j] == 'density':
                            self.data[name]['scdata'].scatter_gene_expression(genes, fig=self.fig, ax=self.ax[len(self.ax)-1],
                                                                                                            density=True)
                        else:
                            self.data[name]['scdata'].scatter_gene_expression(genes, fig=self.fig, ax=self.ax[len(self.ax)-1],
                                                                                                            color=colorSelection[j])
                        self.ax[len(self.ax)-1].set_title(name + ' (color = ' + colorSelection[j] + ')')
                        self.ax[len(self.ax)-1].set_xlabel(' '.join(genes[0]))
                        self.ax[len(self.ax)-1].set_ylabel(' '.join(genes[1]))
                        if len(genes) == 3:
                            self.ax[len(self.ax)-1].set_zlabel(' '.join(genes[2]))
                
                gs.tight_layout(self.fig)
                
                self.tabs.append([tk.Frame(self.notebook), self.fig])
                self.notebook.add(self.tabs[len(self.tabs)-1][0], text=self.plotNameVar.get())

                self.canvas = FigureCanvasTkAgg(self.fig, self.tabs[len(self.tabs)-1][0])
                self.canvas.show()
                self.canvas.get_tk_widget().grid(column=1, row=1, rowspan=10, columnspan=4, sticky='NSEW')

                self.fileMenu.entryconfig(5, state='normal')

                if len(zSelection[0]) > 0 and len(zSelection) == len(xSelection):
                        for ax in self.ax:
                            ax.mouse_init()
                self.currentPlot = 'scatter'
            else:
                print('Error: must select at least one gene for x and y. x and y must also have the same number of selections.')

    def getScatterSelection(self, plot_type='', options=None):
        #popup menu for scatter plot selections
        self.scatterSelection = tk.Toplevel()
        self.scatterSelection.title("Scatter plot options")

        tk.Label(self.scatterSelection, text="For plotting axes, specify a single gene, diffusion component(DC#),"+
                                             " or PCA component(PC#) or a comma separated list (number of items in"+
                                             " each list must be equal). The z-axis is optional", wraplength=500).grid(row=0, column=0, rowspan=2, columnspan=2)
        tk.Label(self.scatterSelection, text="A plot can be colored by a gene, diffusion component(DC#),"+ 
                                             " PCA component(PC#), or \"density\" for kernel density. " +
                                             "The color can also be a solid color(eg: \"blue\").", wraplength=500).grid(row=2, column=0, rowspan=2, columnspan=2)
        #plot name
        tk.Label(self.scatterSelection, text=u"Plot name:").grid(row=4, column=0)
        self.plotNameVar = tk.StringVar()
        self.plotNameVar.set('Plot ' + str(len(self.tabs)))
        tk.Entry(self.scatterSelection, textvariable=self.plotNameVar).grid(row=4, column=1)

        #x
        if plot_type == 'tsne':
            tk.Label(self.scatterSelection, text=u"x: tSNE1", fg="black",bg="white").grid(row=5, column=0)
        elif plot_type in ['PCA', 'Diffusion components']:
            tk.Label(self.scatterSelection, text=u"x:", fg="black", bg="white").grid(row=5, column=0)
            self.xVar = tk.StringVar()
            self.xVar.set(options[0])
            tk.OptionMenu(self.scatterSelection, self.xVar, *options).grid(row=5, column=1)
        else:
            tk.Label(self.scatterSelection, text=u"x:", fg="black",bg="white").grid(row=5, column=0)
            self.xVar = tk.StringVar()
            tk.Entry(self.scatterSelection, textvariable=self.xVar).grid(column=1,row=5)

        #y
        if plot_type == 'tsne':
            tk.Label(self.scatterSelection, text=u"y: tSNE2", fg="black",bg="white").grid(row=6, column=0)
        elif plot_type in ['PCA', 'Diffusion components']:
            tk.Label(self.scatterSelection, text=u"y:", fg="black", bg="white").grid(row=6, column=0)
            self.yVar = tk.StringVar()
            self.yVar.set(options[0])
            tk.OptionMenu(self.scatterSelection, self.yVar, *options).grid(row=6, column=1)
        else:
            tk.Label(self.scatterSelection, text=u"y:", fg="black",bg="white").grid(row=6, column=0)
            self.yVar = tk.StringVar()
            tk.Entry(self.scatterSelection, textvariable=self.yVar).grid(column=1,row=6)

        #z
        if plot_type != 'tsne':
            self.zVar = tk.StringVar()
            if plot_type in ['PCA', 'Diffusion components']:
                tk.Label(self.scatterSelection, text=u"z:", fg="black", bg="white").grid(row=7, column=0)
                self.zVar.set('None')
                tk.OptionMenu(self.scatterSelection, self.zVar, 'None', *options).grid(row=7, column=1)
            else:
                tk.Label(self.scatterSelection, text=u"z:", fg="black",bg="white").grid(row=7, column=0)
                tk.Entry(self.scatterSelection, textvariable=self.zVar).grid(column=1,row=7)

        #color
        tk.Label(self.scatterSelection, text=u"color:", fg="black",bg="white").grid(row=8, column=0)
        self.colorVar = tk.StringVar()
        self.colorVar.set('blue')
        tk.Entry(self.scatterSelection, textvariable=self.colorVar).grid(column=1,row=8)

        tk.Button(self.scatterSelection, text="Plot", command=self.scatterSelection.destroy).grid(column=1, row=9)
        tk.Button(self.scatterSelection, text="Cancel", command=self._cancelScatter).grid(column=0, row=9)
        self.wait_window(self.scatterSelection)

    def _cancelScatter(self):
        self.colorVar.set('')
        self.scatterSelection.destroy()

    def savePlot(self):
        tab = self.notebook.index(self.notebook.select())
        default_name = self.notebook.tab(self.notebook.select(), "text")

        self.plotFileName = filedialog.asksaveasfilename(title='Save Plot', defaultextension='.png', 
                                                         initialfile=default_name)
        if self.plotFileName != None:
            self.tabs[tab][1].savefig(self.plotFileName)

    def saveCSV(self, scdata, col):
        toSave = scdata.tsne.assign(com=pd.Series(col).values)
        self.csvFile = filedialog.asksaveasfile(title='Save as CSV', defaultextension='.csv', mode='w')

        if self.csvFile:
            clusters = []
            cell_map = {}
            for index, row in toSave.iterrows():
                if row['com'] in clusters:
                    cell_map[row['com']].append(index)
                else:
                    clusters.append(row['com'])
                    cell_map[row['com']] = [index]

            clusters = sorted(clusters)

        writer = csv.writer(self.csvFile, delimiter=',')
        writer.writerow(clusters)
        for clus in clusters:
            writer.writerow(cell_map[clus])

        self.csvFile.close()

    def closeCurrentTab(self):
        tab = self.notebook.index(self.notebook.select())
        self.notebook.forget(tab)
        del self.tabs[tab]

    def quitMAGIC(self):
        self.quit()
        self.destroy()

def launch():
    app = magic_gui(None)
    if platform.system() == 'Darwin':
        app.focus_force()
    elif platform.system() == 'Windows':
        app.lift()
        app.call('wm', 'attributes', '.', '-topmost', True)
        app.after_idle(app.call, 'wm', 'attributes', '.', '-topmost', False)        
    elif platform.system() == 'Linux':
        app.focus_force()

    app.title('MAGIC')

    while True:
        try:
            app.mainloop()
            break
        except UnicodeDecodeError:
            pass

if __name__ == "__main__":
    launch()
