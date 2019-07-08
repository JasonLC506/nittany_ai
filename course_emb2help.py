import numpy as np
import cPickle
from scipy.sparse import csr_matrix
import time

embd_file_name = "./result/embds_epoch004"

with open(embd_file_name, "rb") as df:
    embds = cPickle.load(df)

FILTER = 0.8

def emb2help(embds):
    """
    turn course embeddings to course helpfulness,
    here a no course dependent version is given
    :param embds:
    :return:
    """
    C, K = embds.shape
    C = C - 1                       # bias embedding takes one place
    helpful_graph = np.zeros([C, C], dtype=np.float64)
    start = time.time()
    for c_t in range(C):
        for c_s in range(C):
            if c_s == c_t:
                helpful_graph[c_t, c_s] = 0.0
            else:
                helpful_graph[c_t, c_s] = emb2help_single(embds[c_t], embds[c_s], embds[-1])
        if (c_t + 1) % 100 == 0:
            end = time.time()
            print("done %d courses in %s seconds" % (c_t, str(end - start)))
    return helpful_graph


def emb2help_single(emb_c_t, emb_c_s, emb_bias=None):
    if emb_bias is None:
        emb_bias = np.zeros_like(emb_c_s)                        # default for bias embedding
    diff = emb_c_t - np.maximum(emb_c_s, emb_bias)
    # filtering harder course helpfulness #
    if np.sum((emb_c_t - emb_c_s)) < 0.0:
        return 0.0
    diff = diff.clip(min=0.0)
    help = 1.0 / (1.0 + np.sum(diff, axis=-1))
    if help < FILTER:
        help = 0.0
    return help


if __name__ == "__main__":
    helpful_graph = emb2help(embds)
    helpful_hist = np.histogram(helpful_graph, bins=20)
    print helpful_hist
    helpful_graph_sparse = csr_matrix(helpful_graph)
    # with open(embd_file_name + "_help", "wb") as of:
    #     cPickle.dump(helpful_graph_sparse, of)
