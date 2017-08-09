import sys
sys.path.insert(0, '/Users/vincentliu/PycharmProjects/scras/src/scras')
import scras as scras
from rpy2 import robjects as robj
from rpy2.robjects import pandas2ri
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import os
import time

start = time.time()
print("constructing SCData object...")
scdata = scras.SCData.from_csv("/Users/vincentliu/Desktop/Pe'er Lab/Summer 2017/Data/pbmc_4k_dense.csv", "test")
datacopy = copy.deepcopy(scdata)

scdata.normalize_scseq_data()
scdata.log_transform_scseq_data()
datacopy.normalize_scseq_data()
datacopy.log_transform_scseq_data()

print("construct csv string...")
df = datacopy.data
print(df)
to_write = df.to_csv(header=True)
to_write = to_write[1:]
print("writing expression matrix to file...")
# os.getcwd()
data_out = "/Users/vincentliu/Desktop/scras_temp_data.csv"
with open(data_out, 'w') as out_file:
    out_file.write(to_write)

print("running phenograph...")
scpca = scdata.run_pca(n_components=30)
communities, Q = scpca.run_phenograph()
counts = np.bincount(communities)

clustered = scdata.data.assign(communities=pd.Series(communities).values)
target = clustered.loc[clustered['communities'] == 2]
others = clustered.loc[clustered['communities'] != 2]
clustered = pd.concat([target, others])

print("writing community data to file...")
com_out = "/Users/vincentliu/Desktop/scras_temp_com.csv"
with open(com_out, 'w') as out_file:
    for i in communities:
        out_file.write(str(i) + '\n')

print("r running environment setup...")
pandas2ri.activate()
r = robj.r
rscript_path = "/Users/vincentliu/Desktop/Pe'er Lab/Summer 2017/R/MAST-copy.R"
scseq_path = "\"" + data_out + "\""
cluster_path = "\"" + com_out + "\""
r("scseq_path <- " + scseq_path)
r("cluster_path <- " + cluster_path)
print(counts)
largecom = input("which cluster: ")
r("targetcluster <- " + "\"" + "cluster" + str(largecom) + "\"")

print("running r script...")
r.source(rscript_path)
result = r['fcHurdleSig']
result = pandas2ri.ri2py(result)
print(result)

end = time.time()
print(start-end)

"""
hotgenes = list(result['primerid'])
toplot = clustered[hotgenes]
print(toplot)
"""