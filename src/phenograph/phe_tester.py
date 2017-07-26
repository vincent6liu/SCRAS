import phenograph
import numpy as np
import pandas as pd
import magic
import matplotlib
import matplotlib.pyplot as plt

# represent data as a SCData object and automatically normalizes the data
print("reading data from csv file...")
scdata = magic.mg.SCData.from_csv("/Users/vincentliu/Desktop/Pe'er Lab/Summer 2017/Data/pbmc_4k_short.csv")

# log transform the data with default pseudocount 0.1
print("log-transforming the data...")
scdata.log_transform_scseq_data()

# run tsne on the processed data and store the tsne values into a pd dataframe
print("running tSNE on the data...")
scdata.run_tsne()
tsne = scdata.tsne

# run phenograph on the processed data
print("starting PhenoGraph...")
processed = scdata.data
communities, graph, Q = phenograph.cluster(processed, k=15)
toPlot = tsne.assign(com=pd.Series(communities).values)
clusterRec = {}
for index, row in toPlot.iterrows():
    if row['com'] in clusterRec:
        count = clusterRec[row['com']][2]
        new1 = (clusterRec[row['com']][0] * count + row['tSNE1']) / (count+1)
        new2 = (clusterRec[row['com']][1] * count + row['tSNE2']) / (count+1)
        clusterRec[row['com']] = [new1, new2, count+1]
    else:
        clfusterRec[row['com']] = [row['tSNE1'], row['tSNE2'], 1]

# plot the tsne data and color according to the community assignment by phenograph
print("plotting the data...")
fig, ax = plt.subplots()
ax.scatter(tsne['tSNE1'], tsne['tSNE2'], c=communities)
for key in clusterRec:
    x, y = clusterRec[key][0], clusterRec[key][1]
    ax.annotate(str(int(key+2)), (x, y))
plt.show()
