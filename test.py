import cPickle
from scipy.sparse import csr_matrix

helpful = cPickle.load(open("result/embds_epoch004_help", "rb"))
print helpful.nnz