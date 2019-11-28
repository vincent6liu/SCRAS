Single Cell RNA Analysis Suite (SCRAS)
-------------------------------------------------------

SCRAS is implemented in Python3.

#### Installation

        $> git clone git://github.com/vincent6liu/scras.git
        $> cd scras
        $> sudo -H pip3 install .


##### To Use
After following the installation steps listed above, the GUI can be invoked using

        $> python3 src/scras/scras_gui.py 

##### Acknowledgment
SCRAS is implemented based on MAGIC (https://github.com/pkathail/magic) <br />
Clustering is done using PhenoGraph (https://github.com/jacoblevine/PhenoGraph) <br />
Differential expression analysis is done using MAST (https://github.com/RGLab/MAST) <br />
