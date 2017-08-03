from rpy2 import robjects as robj
import rpy2.robjects.packages as rpkgs
from rpy2.robjects import pandas2ri
import rpy2
import pandas as pd
import numpy as np
import sys
import phenograph

sys.path.insert(0, '/Users/vincentliu/PycharmProjects/magic/src/magic')
import mg_new as mg

pandas2ri.activate()
r = robj.r

rscript_path = "/Users/vincentliu/Desktop/Pe'er Lab/Summer 2017/R/MAST-copy.R"
scseq_path = "\"/Users/vincentliu/Desktop/Pe'er Lab/Summer 2017/Data/pbmc_4k_short_R.csv\""
cluster_path = "\"/Users/vincentliu/Desktop/Pe'er Lab/Summer 2017/Data/pbmc4k_short_cluster.csv\""
r("scseq_path <- " + scseq_path)
r("cluster_path <- " + cluster_path)

"""
df = pd.DataFrame.from_csv("/Users/vincentliu/Desktop/Pe'er Lab/Summer 2017/Data/pbmc_4k_short_R.csv")
df = pandas2ri.py2ri_pandasdataframe(df)
cdata = pd.DataFrame.from_csv("/Users/vincentliu/Desktop/Pe'er Lab/Summer 2017/Data/pbmc4k_short_cluster.csv",
                              header=None)
cdata = pandas2ri.py2ri_pandasdataframe(cdata)
rdf = r("df")
rdf = df
rcdata = r("cdata")
rcdata = cdata
"""

print("running R script...")
r.source(rscript_path)
result = r['fcHurdleSig']
result = pandas2ri.ri2py(result)
print(type(result))
print(result)

print("success")