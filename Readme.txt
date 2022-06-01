This reportory contains the python scripts of Spectral Cross-Domain Neural Network (SCDNN) and its application on PTX-XL dataset for ECG time series classification

In this paper we proposed a novel deep learning model named Spectral Cross-domain neural network (SCDNN) with a new block called Soft-adaptive threshold spectral enhancement (SATSE), to simultaneously reveal the key information embedded in spectral and time domains inside the neural network. More precisely, the domain-cross information is captured by a general Convolutional neural network (CNN) backbone, and different information sources are merged by a self-adaptive mechanism to mine the connection between time and spectral domains. The proposed SCDNN is tested with several classification tasks implemented on the public ECG databases PTB-XL and MIT-BIH. SCDNN outperforms the state-of-the-art approaches with a low computational cost regarding a variety of metrics in all classification tasks on both databases, by finding appropriate domains from the infinite spectral mapping.

The Adam.py file contains the modified Adam optimizer for complex values

The training is perfomed in train_temp.py

The PTB-XL dataset can be found at https://www.nature.com/articles/s41597-020-0495-6

The MIT-BIH dataset can be found at https://physionet.org/content/mitdb/1.0.0/
